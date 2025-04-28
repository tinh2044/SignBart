import torch
from torch import nn
import torch.utils.checkpoint
from encoder import Encoder
from decoder import Decoder
from layers import Projection, ClassificationHead


class SignBart(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.joint_idx = config['joint_idx']

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        self.classification_head = ClassificationHead(config['d_model'], config['num_labels'], config['classifier_dropout'])
        self.projection = Projection(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, keypoints, attention_mask, labels=None):

        b = keypoints.shape[0]
        keypoints = keypoints[:, :, self.joint_idx, :]
        x_embed, y_embed = self.projection(keypoints)

        encoder_outputs = self.encoder(x_embed=x_embed, attention_mask=attention_mask)

        decoder_attention_mask = attention_mask
        
        decoder_outputs = self.decoder(
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            attention_mask=decoder_attention_mask,
            y_embed=y_embed)

        
        last_indices = (decoder_attention_mask == 1).float().cumsum(dim=1).argmax(dim=1)

        last_decoder_outputs = decoder_outputs[torch.arange(b), last_indices, :]

        logits = self.classification_head(last_decoder_outputs)

        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.compute_loss(logits, labels)
        else:
            loss = None

        return (loss, logits)

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)


if __name__ == "__main__":
    import yaml
    with open("./configs/LSA-64.yaml") as f:
        config = yaml.safe_load(f)
    model = SignBart(config)

    old_s = model.encoder.embed_positions.state_dict()['weight'].clone()
    
    pretrained_path = "./pretrained_models/LSA-64.pth"
        
    print(f"Load checkpoint from file : {pretrained_path}")
    state_dict = torch.load(pretrained_path)
    ret = model.load_state_dict(state_dict, strict=False)
    
    print("Missing keys: ", ret.missing_keys)
    print("Unexpected keys: ", ret.unexpected_keys)
    
    new_s = model.encoder.embed_positions.state_dict()['weight'].clone()
    
    print(torch.equal(old_s, new_s))
