import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys
sys.path.append('/home/sunner/Save/coco/PythonAPI')

cap = dset.CocoCaptions(root='./data/', \
    annFile = './ann/',
    transform=transforms.ToTensor())
img, target = cap[:-1]
print(img.size())