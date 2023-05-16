import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from .basenet import *
from .baselines import *
from ..utils import *


class Res18CropEncoder(nn.Module):
    def __init__(self, resnet, CNN_embed_dim=256):
        super().__init__()
        
        self.resnet = resnet
        self.fc = nn.Linear(512, CNN_embed_dim)

    @torch.no_grad()    
    def forward(self, x_5d, x_lengths):
        x_seq = []
        batch_size = x_5d.size(0)
        for i in range(batch_size):
            cnn_embed_seq = []
            for t in range(x_lengths[i]):
                img = x_5d[i, t, :, :, :]
                x = self.resnet(torch.unsqueeze(img,dim=0))  # ResNet
                x = self.fc(x)
                x = F.relu(x)
                x = x.view(x.size(0), -1) # flatten output of conv
                cnn_embed_seq.append(x)                    
            # swap time and sample dim such that (sample dim=1, time dim, CNN latent dim)
            embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
            embed_seq = torch.squeeze(embed_seq)
            fea_dim = embed_seq.shape[-1]
            embed_seq = embed_seq.view(-1,fea_dim)
            x_seq.append(embed_seq)
        
        x_padded = nn.utils.rnn.pad_sequence(x_seq,batch_first=True, padding_value=0)
        return x_padded
    
class MobilenetCropEncoder(nn.Module):
    def __init__(self, mobilenet, CNN_embed_dim=256):
        super().__init__()
        
        self.mobilenet = mobilenet
        # in_features: get the input size of the classifier layer of original mobilenet v3
        in_features = self.mobilenet.classifier[0].in_features
        print('in_features: ', in_features, flush=True)
        self.mobilenet.classifier = torch.nn.Identity()
        self.fc = nn.Linear(in_features, CNN_embed_dim)

    @torch.no_grad()    
    def forward(self, x_5d, x_lengths):
        x_seq = []
        batch_size = x_5d.size(0)
        for i in range(batch_size):
            cnn_embed_seq = []
            for t in range(x_lengths[i]):
                img = x_5d[i, t, :, :, :]
                x = self.mobilenet(torch.unsqueeze(img,dim=0))  # MobileNet
                x = self.fc(x)
                x = F.relu(x)
                x = x.view(x.size(0), -1) # flatten output of conv
                cnn_embed_seq.append(x)                    
            # swap time and sample dim such that (sample dim=1, time dim, CNN latent dim)
            embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
            embed_seq = torch.squeeze(embed_seq)
            fea_dim = embed_seq.shape[-1]
            embed_seq = embed_seq.view(-1,fea_dim)
            x_seq.append(embed_seq)
        
        x_padded = nn.utils.rnn.pad_sequence(x_seq,batch_first=True, padding_value=0)
        return x_padded
        
        
class Res18RoIEncoder(nn.Module):
    """
    CNN-encoder with ResNet-18 backbone
    Input: a sequence of  RGB images Tx3xHxW (0<T<T_max)
    Output: T_max x 1 x CNN_embed_dim feature vector
    """
    def __init__(self, resnet, CNN_embed_dim=256):
        super().__init__()
        
        self.resnet = resnet
        self.fc = nn.Linear(1024, CNN_embed_dim)
        
    @torch.no_grad()
    def forward(self, x_5d, x_lengths):
        x_seq = []
        for i in range(x_5d.size(0)):
            cnn_embed_seq = []
            for t in range(x_lengths[i]):
                img = x_5d[i, t, :, :, :]
                x = self.resnet(torch.unsqueeze(img,dim=0))  # ResNet
                x = self.fc(x)
                x = F.relu(x)
                x = x.view(x.size(0), -1) # flatten output of conv
                cnn_embed_seq.append(x)                    
            # swap time and sample dim such that (sample dim=1, time dim, CNN latent dim)
            embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
            embed_seq = torch.squeeze(embed_seq)
            fea_dim = embed_seq.shape[-1]
            embed_seq = embed_seq.view(-1,fea_dim)
            x_seq.append(embed_seq)
        
        x_padded = nn.utils.rnn.pad_sequence(x_seq,batch_first=True, padding_value=0)
        
        
        return x_padded


class DecoderRNN_IMBS(nn.Module):
    def __init__(self, CNN_embeded_size=256, h_RNN_layers=1, h_RNN_0=256, h_RNN_1=64,
                 h_RNN_2=16, h_FC0_dim=128, h_FC1_dim=64, h_FC2_dim=86, drop_p=0.2):
        super().__init__()
        self.CNN_embeded_size= CNN_embeded_size
        self.h_RNN_0 = h_RNN_0
        self.h_RNN_1 = h_RNN_1
        self.h_RNN_2 = h_RNN_2 # RNN hidden nodes
        self.h_FC0_dim = h_FC0_dim
        self.h_FC1_dim = h_FC1_dim
        self.h_FC2_dim = h_FC2_dim
    
        # image feature decoder
        self.RNN_0 = nn.LSTM(
            input_size=self.CNN_embeded_size,
            hidden_size=self.h_RNN_0,        
            num_layers=1,       
            batch_first=True,       #  (batch, time_step, input_size)
        )
        # motion decoder
        self.RNN_1 = nn.LSTM(
            input_size=8,
            hidden_size=self.h_RNN_1,        
            num_layers=h_RNN_layers,       
            batch_first=True,       #  (batch, time_step, input_size)
        )
        # behavior  decoder
        # keep size at 4 (since we delete c/nc behaviorm but add action)
        self.RNN_2 = nn.LSTM(
            input_size=4,
            hidden_size=self.h_RNN_2,        
            num_layers=h_RNN_layers,       
            batch_first=True,       #  (batch, time_step, input_size)
        )
        self.fc0 = nn.Linear(self.h_RNN_0, self.h_FC0_dim)
        self.fc1 = nn.Linear(self.h_RNN_1 + self.h_FC0_dim, self.h_FC1_dim)
        self.dropout = nn.Dropout(p=drop_p)
        # change +6 to +5 since we delete one element(motion description) in scene descriptions
        self.fc2 = nn.Linear(self.h_FC1_dim + self.h_RNN_2 + 5, self.h_FC2_dim)
        self.fc3 = nn.Linear(self.h_FC2_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, xc_3d, xp_3d, xb_3d, xs_2d, x_lengths):  
        # print("entered decoder forward", flush=True)      
        # use input of descending length
        # print("xc_3d shape", xc_3d.shape, "xp_3d shape", xp_3d.shape, 'xb_3d shape', xb_3d.shape, flush=True)
        packed_x0_RNN = torch.nn.utils.rnn.pack_padded_sequence(xc_3d, x_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        packed_x1_RNN = torch.nn.utils.rnn.pack_padded_sequence(xp_3d, x_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        packed_x2_RNN = torch.nn.utils.rnn.pack_padded_sequence(xb_3d, x_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        self.RNN_0.flatten_parameters()
        self.RNN_1.flatten_parameters()
        self.RNN_2.flatten_parameters()

        packed_RNN_out_0, _ = self.RNN_0(packed_x0_RNN, None)
        # print("RNN0 done", flush=True)
        packed_RNN_out_1, _ = self.RNN_1(packed_x1_RNN, None)
        # print("RNN1 done", flush=True)
        packed_RNN_out_2, _ = self.RNN_2(packed_x2_RNN, None)
        # print("RNN2 done", flush=True)
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        RNN_out_0, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_0, batch_first=True)
        RNN_out_0 = RNN_out_0.contiguous()
        RNN_out_1, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_1, batch_first=True)
        RNN_out_1 = RNN_out_1.contiguous()
        RNN_out_2, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_2, batch_first=True)
        RNN_out_2 = RNN_out_2.contiguous()
      
        # print(f'passed RNN encoding', flush=True)
    
        # choose RNN_out at the last time step
        output_0 = RNN_out_0[:, -1, :]
        output_1 = RNN_out_1[:, -1, :]
        output_2 = RNN_out_2[:, -1, :]
        
        # 
        x0 = self.fc0(output_0)
        x0 = F.relu(x0)
        x0 = self.dropout(x0)
        x_ipv = torch.cat((x0, output_1), dim=1)
        x1 = self.fc1(x_ipv)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        # print('passed fc1', flush=True)
        x_ipvb = torch.cat((x1, output_2, xs_2d), dim=1)
        x = self.fc2(x_ipvb)
        x = F.relu(x)
        # print('passed fc2', flush=True)
        x = self.fc3(x)
        x = self.act(x)
        # print('passed fc3', flush=True)
        return x

def build_encoder_res18(args):
    """
    Construct CNN encoder with resnet-18 backbone

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.encoder_type == 'CC': #use crop-context encoder
        if args.encoder_pretrained:
             res18 = ResnetBlocks(torchvision.models.resnet18(pretrained=False))
             res_backbone = res18.build_backbone(use_pool=True, use_last_block=True, pifpaf_bn=False)
             res18_cc_cpu = Res18Crop(backbone=res_backbone)
             res18_cc_gpu = res18_cc_cpu.to(device)
             checkpoint_path = args.path_to_encoder # change it here to load your fine-tuned encoder
             _ = load_from_checkpoint(checkpoint_path, res18_cc_gpu, optimizer=None, scheduler=None, verbose=True)
             # remove fc
             res_modules = list(res18_cc_gpu.children())[:3]  # delete the last fc layer.
             cnn_gpu = nn.Sequential(*res_modules)

        else:
            if args.mobilenetsmall:
                print('Using mobilenetv3 small as cnn encoder!!')
                # small mobilev3 model
                mobilev3_cpu = torchvision.models.mobilenet_v3_small(pretrained=True)
                cnn_gpu = mobilev3_cpu.to(device)
            elif args.mobilenetbig:
                print('Using mobilenetv3 big as cnn encoder!!')
                # big mobilev3 model
                mobilev3_cpu = torchvision.models.mobilenet_v3_large(pretrained=True)
                cnn_gpu = mobilev3_cpu.to(device)
            else:
                print('Using resnet18 cnn encoder!!')
                res18_cpu = torchvision.models.resnet18(pretrained=True)
                # remove last fc
                res18_cpu.fc = torch.nn.Identity()
                cnn_gpu = res18_cpu.to(device)

        if args.mobilenetsmall or args.mobilenetbig:
            encoder_cnn = MobilenetCropEncoder(mobilenet=cnn_gpu)
        else:
            encoder_cnn = Res18CropEncoder(resnet=cnn_gpu)
    else:
         if args.encoder_pretrained:
             res18 = ResnetBlocks(torchvision.models.resnet18(pretrained=False))
             res_till_4 = res18.build_backbone(use_pool=False, use_last_block=False, pifpaf_bn=False)
             last_block = res18.block5()
             res18_roi_cpu = Res18RoI(res_till_4, last_block)
             res18_roi_gpu = res18_roi_cpu.to(device)
             checkpoint_path = args.path_to_encoder # change it here
             _ = load_from_checkpoint(checkpoint_path, res18_roi_gpu, optimizer=None, scheduler=None, verbose=True)
         else:
             res18 = ResnetBlocks(torchvision.models.resnet18(pretrained=True))
             res_till_4 = res18.build_backbone(use_pool=False, use_last_block=False, pifpaf_bn=False)
             last_block = res18.block5()
             res18_roi_cpu = Res18RoI(res_till_4, last_block)
             res18_roi_gpu = res18_roi_cpu.to(device)
         # remove fc
         res18_roi_gpu.FC = torch.nn.Identity()
         res18_roi_gpu.dropout = torch.nn.Identity()
         res18_roi_gpu.act = torch.nn.Identity()
         # encoder
         encoder_cnn = Res18RoIEncoder(encoder=res18_roi_gpu)
    encoder_cnn.to(device)
    return encoder_cnn


