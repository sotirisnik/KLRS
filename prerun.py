import tensorflow as tf
import tensorflow_datasets as tfds
import torchvision.models as models

for dataset_name in [ "mnist", "fashion_mnist", "cifar10", "cifar100" ]:
    ds = tfds.load( dataset_name, as_supervised=True, batch_size=-1 )
    print( dataset_name, "OK")

pytorch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
print( 'resnet ok' )