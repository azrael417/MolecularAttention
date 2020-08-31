import torch.nn as nn
import torchvision.models as models
import torch
class ImageModel(nn.Module):

    def __init__(self, intermediate_rep=256,  nheads=1, outs=1, dr=0, classifacation=False, linear_layers=2, model_path=None, pretrain=True):
        super(ImageModel, self).__init__()
        self.return_attn = True
        self.outs = outs
        self.nheads = nheads


        if model_path is None:
            resnet18 = models.resnet101(pretrained=pretrain)
        else:
            resnet18 = models.resnet101(pretrained=False)
            if pretrain:
                resnet18.load_state_dict(torch.load(model_path))

        resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

        if self.nheads > 0:
            self.resnet181 = nn.Sequential(*list(resnet18.children())[:5])
            self.resnet182 = nn.Sequential(*list(resnet18.children())[5:])
            self.attention = nn.Sequential(
                nn.Conv2d(256, nheads, kernel_size=5, padding=2, stride=1),
            )
        else:
            self.resnet181 = nn.Sequential(*list(resnet18.children()))

        self.model = nn.Sequential(
            nn.Linear(2048, intermediate_rep),
            nn.BatchNorm1d(intermediate_rep),
            nn.ReLU(),
            nn.Dropout(dr),
        )

        self.linears = nn.ModuleList()
        for i in range(linear_layers):
            self.linears.append(nn.Linear(intermediate_rep, intermediate_rep))
            self.linears.append(nn.ReLU(),)
            self.linears.append(nn.Dropout(dr))

        self.linear = nn.Sequential(*self.linears)

        self.prop_model = nn.Sequential(
            self.linear,

            nn.Linear(intermediate_rep, intermediate_rep),
            nn.ReLU(),
            nn.Dropout(dr),

            nn.Linear(intermediate_rep, self.outs)
        )


    def forward(self, features):
        if self.nheads > 0:
            image = self.resnet181(features)
            attention = self.attention(image)
            attention = nn.functional.softmax(attention.view(attention.shape[0], self.nheads, -1), dim=2).view(attention.shape)
            image = self.resnet182(image * attention)
            image = image.view(features.shape[0], -1)
        else:
            image = self.resnet181(features).view(features.shape[0], -1)
            self.return_attn = False

        if self.return_attn:
            return self.prop_model(self.model(image)), attention
        else:
            return self.prop_model(self.model(image))
