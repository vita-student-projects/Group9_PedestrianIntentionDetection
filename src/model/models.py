import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .basenet import *
from .baselines import *
from ..utils import *


class CNNEncoder(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.activation = F.relu if activation == 'relu' else F.sigmoid

    def freeze_backbone(self,n_layer=None):
        total_layers=len(list(self.backbone.children()))
        if n_layer is None: n_layer=total_layers
        for child in list(self.backbone.children())[:n_layer]:
            for para in child.parameters():
                para.requires_grad = False
        print(f"freeze {n_layer} layers out of {total_layers} layers")

    def turn_off_running_stats(self):
        
        def _turn_off_running_stats_recursive(module):    
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                return
            for child in module.children():
                _turn_off_running_stats_recursive(child)

        _turn_off_running_stats_recursive(self.backbone)


    def forward(self, x_5d, x_lengths):
        x_seq = []
        batch_size = x_5d.size(0)
        for i in range(batch_size):
            cnn_embed_seq = []
            for t in range(x_lengths[i]):
                img = x_5d[i, t, :, :, :]
                x = self.backbone(torch.unsqueeze(img,dim=0))  
                x = self.fc(x)
                x = self.activation(x)
                x = x.view(x.size(0), -1) # flatten output of conv
                cnn_embed_seq.append(x)                    
            # swap time and sample dim such that (sample dim=1, time dim, CNN latent dim)
            embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
            embed_seq = torch.squeeze(embed_seq, 1)
            fea_dim = embed_seq.shape[-1]
            embed_seq = embed_seq.view(-1,fea_dim)
            x_seq.append(embed_seq)
        
        x_padded = nn.utils.rnn.pad_sequence(x_seq,batch_first=True, padding_value=0)
        return x_padded


class Res18CropEncoder(CNNEncoder):
    def __init__(self, resnet, CNN_embed_dim=256, activation='relu'):
        super().__init__(activation=activation)
        self.backbone = resnet
        self.fc = nn.Linear(512, CNN_embed_dim)
    

class Res18Classifier(CNNEncoder):
    def __init__(self, CNN_embed_dim=256, activation='relu'):
        super().__init__(activation=activation)
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, CNN_embed_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(CNN_embed_dim, 1),
        )


class MobilenetCropEncoder(CNNEncoder):
    def __init__(self, mobilenet, CNN_embed_dim=256, activation='relu'):
        super().__init__(activation=activation)
        self.backbone = mobilenet
        # in_features: get the input size of the classifier layer of original mobilenet v3
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = torch.nn.Identity()
        self.fc = nn.Linear(in_features, CNN_embed_dim)


class RNNClassifier(nn.Module):
    def __init__(self, input_size, rnn_embeding_size=256, classification_head_size=128, drop_p=0.5, h_RNN_layers=1):
        super().__init__()
        self.threshold = 0.5
    
        self.RNN = nn.LSTM(
            input_size=input_size,
            hidden_size=rnn_embeding_size,        
            num_layers=h_RNN_layers,       
            batch_first=True,       #  (batch, time_step, input_size)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(rnn_embeding_size, classification_head_size),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(classification_head_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_seq, seq_lengths):  
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input_seq, seq_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        self.RNN.flatten_parameters()
        packed_RNN_out, _ = self.RNN(packed_inputs, None)
        RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out[:, -1, :].contiguous()
        pred = self.classification_head(RNN_out).unsqueeze(-1)
        return pred


class CRNNClassifier(nn.Module):
    def __init__(self, pos_vel_embedding_size, cnn_embedding_size, rnn_embeding_size=256, classification_head_size=128, drop_p=0.5, h_RNN_layers=1):
        super().__init__()
    
        res18= torchvision.models.resnet18(pretrained=True)
        res18.fc = torch.nn.Identity()
        self.cnn_encoder = Res18CropEncoder(resnet=res18, CNN_embed_dim=cnn_embedding_size)

        self.image_rnn = nn.LSTM(
            input_size=cnn_embedding_size,
            hidden_size=rnn_embeding_size,        
            num_layers=h_RNN_layers,       
            batch_first=True, 
        )

        self.position_rnn = nn.LSTM(
            input_size=pos_vel_embedding_size,
            hidden_size=rnn_embeding_size,
            num_layers=h_RNN_layers,
            batch_first=True,
        )

        self.classification_head = nn.Sequential(
            nn.Linear(2 * rnn_embeding_size, classification_head_size),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(classification_head_size, 1),
            nn.Sigmoid()
        )

    def from_pretrained(self, cnn_encoder_path, position_velocity_rnn_path):
        self.cnn_encoder.load_state_dict(torch.load(cnn_encoder_path)['encoder_state_dict'])
        self.position_rnn.load_state_dict(torch.load(position_velocity_rnn_path)['decoder_state_dict'])

    def forward(self, image_seq, pos_vel_seq, seq_lengths):  
        padded_image_inputs = self.cnn_encoder(image_seq, seq_lengths)

        packed_image_inputs = torch.nn.utils.rnn.pack_padded_sequence(padded_image_inputs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_pos_vel_inputs = torch.nn.utils.rnn.pack_padded_sequence(pos_vel_seq, seq_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        
        self.image_rnn.flatten_parameters()
        self.position_rnn.flatten_parameters()

        packed_image_rnn_out, _ = self.image_rnn(packed_image_inputs, None)
        packed_pos_vel_rnn_out, _ = self.position_rnn(packed_pos_vel_inputs, None)

        image_rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_image_rnn_out, batch_first=True)
        pos_vel_rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_pos_vel_rnn_out, batch_first=True)

        combined_out = torch.cat((image_rnn_out, pos_vel_rnn_out), dim=-1)[:, -1, :].contiguous()
        pred = self.classification_head(combined_out)
        return pred


class RNNClassifier(nn.Module):
    def __init__(self, input_size, rnn_embeding_size=256, classification_head_size=128, drop_p=0.2, h_RNN_layers=1):
        super().__init__()
        self.threshold = 0.5
    
        self.RNN = nn.LSTM(
            input_size=input_size,
            hidden_size=rnn_embeding_size,        
            num_layers=h_RNN_layers,       
            batch_first=True,       #  (batch, time_step, input_size)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(rnn_embeding_size, classification_head_size),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(classification_head_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_seq, seq_lengths):  

        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input_seq, seq_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        # TODO: why?
        self.RNN.flatten_parameters()
        packed_RNN_out, _ = self.RNN(packed_inputs, None)
        RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out[:, -1, :].contiguous()
        pred = self.classification_head(RNN_out).unsqueeze(-1)
        return pred

class RNNClassifier(nn.Module):
    def __init__(self, input_size, rnn_embeding_size=256, classification_head_size=128, drop_p=0.2, h_RNN_layers=1):
        super().__init__()
        self.threshold = 0.5
    
        self.RNN = nn.LSTM(
            input_size=input_size,
            hidden_size=rnn_embeding_size,        
            num_layers=h_RNN_layers,       
            batch_first=True,       #  (batch, time_step, input_size)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(rnn_embeding_size, classification_head_size),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(classification_head_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_seq, seq_lengths):  

        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input_seq, seq_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        # TODO: why?
        self.RNN.flatten_parameters()
        packed_RNN_out, _ = self.RNN(packed_inputs, None)
        RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out[:, -1, :].contiguous()
        pred = self.classification_head(RNN_out).unsqueeze(-1)
        return pred

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

        packed_x0_RNN = torch.nn.utils.rnn.pack_padded_sequence(xc_3d, x_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        packed_x1_RNN = torch.nn.utils.rnn.pack_padded_sequence(xp_3d, x_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        packed_x2_RNN = torch.nn.utils.rnn.pack_padded_sequence(xb_3d, x_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        self.RNN_0.flatten_parameters()
        self.RNN_1.flatten_parameters()
        self.RNN_2.flatten_parameters()

        # None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) 
        packed_RNN_out_0, _ = self.RNN_0(packed_x0_RNN, None)
        packed_RNN_out_1, _ = self.RNN_1(packed_x1_RNN, None)
        packed_RNN_out_2, _ = self.RNN_2(packed_x2_RNN, None)

        RNN_out_0, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_0, batch_first=True)
        RNN_out_0 = RNN_out_0.contiguous()
        RNN_out_1, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_1, batch_first=True)
        RNN_out_1 = RNN_out_1.contiguous()
        RNN_out_2, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_2, batch_first=True)
        RNN_out_2 = RNN_out_2.contiguous()
    
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
        x_ipvb = torch.cat((x1, output_2, xs_2d), dim=1)
        x = self.fc2(x_ipvb)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.act(x)
        return x

def build_encoder_res18(args, hidden_dim=256, activation='relu'):
    """
    Construct CNN encoder with resnet-18 backbone
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone == 'mobilenetsmall':
        print('Using mobilenetv3 small as cnn encoder!!')
        # small mobilev3 model
        mobilev3_cpu = torchvision.models.mobilenet_v3_small(pretrained=True)
        cnn_gpu = mobilev3_cpu.to(device)
    elif args.backbone == 'mobilenetbig':
        print('Using mobilenetv3 big as cnn encoder!!')
        # big mobilev3 model
        mobilev3_cpu = torchvision.models.mobilenet_v3_large(pretrained=True)
        cnn_gpu = mobilev3_cpu.to(device)
    else:
        print('Using resnet18 cnn encoder!!')
        res18= torchvision.models.resnet18(pretrained=True)
        # remove last fc
        res18.fc = torch.nn.Identity()
        cnn_gpu = res18.to(device)
    if args.backbone in ['mobilenetsmall', 'mobilenetbig']:
        encoder_cnn = MobilenetCropEncoder(mobilenet=cnn_gpu, CNN_embed_dim=hidden_dim, activation=activation)
    else:
        encoder_cnn = Res18CropEncoder(resnet=cnn_gpu, CNN_embed_dim=hidden_dim, activation=activation)
    encoder_cnn.to(device)
    return encoder_cnn


