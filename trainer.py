import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from pathlib import Path

import models
import utils

class GTA():
    def __init__(self, params, src_train_loader, src_val_loader, tgt_loader):
        self.device = params.device
        self.n_epochs = params.n_epochs
        self.batch_size = params.batch_size
        self.n_classes = params.n_classes
        self.adv_weight = params.adv_weight
        self.alpha = params.alpha
        self.output_root = params.output_root

        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader
        self.tgt_loader = tgt_loader
        
        # Define networks
        self.net_F = models.NetF(params)
        self.net_G = models.NetG(params)
        self.net_D = models.NetD(params)
        self.net_C = models.NetC(params)

        # Weight initialization
        self.net_F.apply(utils.init_weights)
        self.net_G.apply(utils.init_weights)
        self.net_D.apply(utils.init_weights)
        self.net_C.apply(utils.init_weights)

        # Define loss criterions
        self.criterion_cls = nn.CrossEntropyLoss(reduction='sum')
        self.criterion_data = nn.BCELoss(reduction='sum')

        # Define optimizers
        self.optim_F = optim.Adam(self.net_F.parameters(), lr=params.lr)
        self.optim_G = optim.Adam(self.net_G.parameters(), lr=params.lr)
        self.optim_D = optim.Adam(self.net_D.parameters(), lr=params.lr)
        self.optim_C = optim.Adam(self.net_C.parameters(), lr=params.lr)

        self.net_F.to(params.device)
        self.net_G.to(params.device)
        self.net_D.to(params.device)
        self.net_C.to(params.device)

        self.model_dir = params.model_dir
        self.vis_dir = params.vis_dir

        self.best_val_loss = float('Inf')
        self.hist_acc = []
        self.hist_loss = []

    def validate(self, epoch, train_loss):
        self.net_F.eval()
        self.net_C.eval()

        tot_corrects = 0
        tot_loss = 0

        n_samples = 0
    
        for i, data in enumerate(self.src_val_loader):
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)

            emb = self.net_F(img)

            output_C = self.net_C(emb)

            _, pred = torch.max(output_C, 1)

            tot_corrects += torch.sum(pred == label).item()

            loss = self.criterion_cls(output_C, label)

            tot_loss += loss.item()

            n_samples += len(label)
            
        print('|__ val acc: {:.3f}, loss: {:.3f}'
            .format(tot_corrects / n_samples, tot_loss / n_samples))
    
        # Save checkpoints
        if epoch % 10 == 0:
            torch.save(self.net_F.state_dict(),
                self.model_dir / 'net_F_e{}.pth'.format(epoch))
            torch.save(self.net_C.state_dict(),
                self.model_dir / 'net_C_e{}.pth'.format(epoch))
        
        # Save best models
        val_loss = train_loss

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

            torch.save(self.net_F.state_dict(),
                self.model_dir / 'net_F_best.pth')
            torch.save(self.net_C.state_dict(),
                self.model_dir / 'net_C_best.pth')

            with open(self.model_dir / 'best_loss.txt', 'w') as f:
                f.write('{}\t{}\n'.format(epoch, val_loss))

            print('|__ best model saved')
            
    def train(self):
        # real_label: original img, fake_label: generated img
        real_label = torch.ones(self.batch_size, device=self.device)
        fake_label = torch.zeros(self.batch_size, device=self.device)
        
        for epoch in range(self.n_epochs):
            print('Epoch {} --------------------'.format(epoch + 1))

            self.net_F.train()    
            self.net_G.train()    
            self.net_D.train()    
            self.net_C.train()

            running_corrects = 0
            running_loss = 0
            tot_corrects = 0
            tot_loss_F = 0
            tot_loss_G = 0
            tot_loss_D = 0
            tot_loss_C = 0
            tot_loss = 0
            n_samples = 0

            for i, (src_data, tgt_data) in enumerate(zip(self.src_train_loader, self.tgt_loader)):
                src_img, src_label = src_data
                tgt_img, _ = tgt_data

                # Scale img to [-1, 1] for generator
                src_img_scaled = (torch.stack([utils.unnormalize(img)
                    for img in src_img.clone()]) - 0.5) * 2

                src_label_onehot = torch.zeros((self.batch_size, self.n_classes + 1
                    )).scatter_(1, src_label.view(-1, 1), 1)
                tgt_label_onehot = torch.zeros((self.batch_size, self.n_classes + 1
                    )).index_fill_(1, torch.tensor([self.n_classes]), 1)
                
                src_img = src_img.to(self.device)
                src_img_scaled = src_img_scaled.to(self.device)
                src_label = src_label.to(self.device)
                tgt_img = tgt_img.to(self.device)
                src_label_onehot = src_label_onehot.to(self.device)
                tgt_label_onehot = tgt_label_onehot.to(self.device)
                
                # Update net D
                self.net_D.zero_grad()

                src_emb = self.net_F(src_img)
                src_emb_n_label = torch.cat((src_emb, src_label_onehot), 1)
                src_gen = self.net_G(src_emb_n_label)

                tgt_emb = self.net_F(tgt_img)
                tgt_emb_n_label = torch.cat((tgt_emb, tgt_label_onehot), 1)
                tgt_gen = self.net_G(tgt_emb_n_label)

                src_orig_output_D_data, src_orig_output_D_cls = self.net_D(src_img_scaled)   
                loss_D_src_data_real = self.criterion_data(src_orig_output_D_data, real_label) 
                loss_D_src_cls_real = self.criterion_cls(src_orig_output_D_cls, src_label) 

                src_gen_output_D_data, src_gen_output_D_cls = self.net_D(src_gen)
                loss_D_src_data_fake = self.criterion_data(src_gen_output_D_data, fake_label)

                tgt_gen_output_D_data, _ = self.net_D(tgt_gen)
                loss_D_tgt_data_fake = self.criterion_data(tgt_gen_output_D_data, fake_label)

                loss_D = (loss_D_src_data_real + loss_D_src_cls_real +
                    loss_D_src_data_fake + loss_D_tgt_data_fake)
                loss_D.backward(retain_graph=True)

                self.optim_D.step()
                
                # Recompute net D outputs after updating net D
                src_gen_output_D_data, src_gen_output_D_cls = self.net_D(src_gen)
                tgt_gen_output_D_data, _ = self.net_D(tgt_gen)

                # Update net G
                self.net_G.zero_grad()

                loss_G_data = self.criterion_data(src_gen_output_D_data, real_label)
                loss_G_cls = self.criterion_cls(src_gen_output_D_cls, src_label)

                loss_G = loss_G_data + loss_G_cls
                loss_G.backward(retain_graph=True)

                self.optim_G.step()
                
                # Update net C
                self.net_C.zero_grad()

                output_C = self.net_C(src_emb)

                loss_C = self.criterion_cls(output_C, src_label)
                loss_C.backward(retain_graph=True)

                self.optim_C.step()
                
                # Update net F
                self.net_F.zero_grad()

                loss_F_src = (self.adv_weight *
                    self.criterion_cls(src_gen_output_D_cls, src_label))

                loss_F_tgt = (self.adv_weight * self.alpha *
                    self.criterion_data(tgt_gen_output_D_data, real_label))
                
                loss_F = loss_C + loss_F_src + loss_F_tgt
                loss_F.backward()

                self.optim_F.step()

                # Record acc & loss
                _, predicts = torch.max(output_C, 1)
                corrects = torch.sum(predicts == src_label)

                running_corrects += corrects.item()
                tot_corrects += corrects.item()
                running_loss += loss_F.item() + loss_G.item() + loss_D.item() + loss_C.item()
                tot_loss += loss_F.item() + loss_G.item() + loss_D.item() + loss_C.item()
                tot_loss_F += loss_F.item()
                tot_loss_D += loss_D.item()
                tot_loss_G += loss_G.item()
                tot_loss_C += loss_C.item()
                n_samples += len(src_label)

                # Print record
                batches_to_print = max(len(self.src_train_loader), len(self.tgt_loader)) // 5
                if i % batches_to_print == batches_to_print - 1:
                    print('| {}/{} acc: {:.3f}, loss: {:.3f}'
                        .format(i + 1, len(self.src_train_loader),
                        running_corrects / batches_to_print / self.batch_size,
                        running_loss / batches_to_print / self.batch_size))

                    running_corrects = 0
                    running_loss = 0
            
            print('|__ train acc: {:.3f}, loss: {:.3f} (F: {:.3f}, D: {:.3f}, G: {:.3f}, C: {:.3f})'
                .format(tot_corrects / n_samples,
                tot_loss / n_samples, tot_loss_F / n_samples, tot_loss_D / n_samples,
                tot_loss_G / n_samples, tot_loss_C / n_samples))

            self.hist_acc.append(tot_corrects / n_samples)
            self.hist_loss.append(tot_loss / n_samples)
                
            # Visualization
            if epoch % 5 == 0:
                torchvision.utils.save_image(
                    src_img_scaled[: 64] / 2 + 0.5,
                    self.vis_dir / 'source_e{}.png'.format(epoch + 1))
                torchvision.utils.save_image(
                    [utils.unnormalize(img.cpu()) for img in tgt_img[: 64]],
                    self.vis_dir / 'target_e{}.png'.format(epoch + 1))
                torchvision.utils.save_image(
                    src_gen[: 64] / 2 + 0.5,
                    self.vis_dir / 'source_gen_e{}.png'.format(epoch + 1))
                torchvision.utils.save_image(
                    tgt_gen[: 64] / 2 + 0.5,
                    self.vis_dir / 'target_gen_e{}.png'.format(epoch + 1))
                    
            # Validate per epoch
            self.validate(epoch + 1, tot_loss / n_samples)

            # Process Visualization
            import matplotlib.pyplot as plt
            x_range = range(1, epoch + 2)
            plt.plot(x_range, self.hist_acc)
            plt.savefig(str(self.vis_dir / 'train_acc'))
            plt.close()
            plt.plot(x_range, self.hist_loss)
            plt.savefig(str(self.vis_dir / 'train_loss'))
            plt.close()