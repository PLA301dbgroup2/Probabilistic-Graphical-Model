import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
	def __init__(self,alpha=0.25,gamma=2):
		super(FocalLoss,self).__init__()
		self.alpha=alpha
		self.gamma=gamma
	
	def forward(self,preds,labels):

		eps=1e-7
		sm = nn.Softmax(dim=1)
		preds = sm(preds)
		probs = preds[:, 0]
		loss_1=-1*self.alpha*torch.pow((1-probs),self.gamma)*torch.log(probs+eps)*labels
		loss_0=-1*(1-self.alpha)*torch.pow(probs,self.gamma)*torch.log(1-probs+eps)*(1-labels)
		loss=loss_0+loss_1
		return torch.mean(loss)
	

class Focal_Loss():
	def __init__(self,weight,gamma=2):
		super(Focal_Loss,self).__init__()
		self.gamma=gamma
		self.weight=weight
	def forward(self,preds,labels):

		eps=1e-7
		y_pred =preds.view((preds.size()[0],preds.size()[1],-1)) #B*C*H*W->B*C*(H*W)
		
		target=labels.view(y_pred.size()) #B*C*H*W->B*C*(H*W)
		
		ce=-1*torch.log(y_pred+eps)*target
		floss=torch.pow((1-y_pred),self.gamma)*ce
		floss=torch.mul(floss,self.weight)
		floss=torch.sum(floss,dim=1)
		return torch.mean(floss)