import jax
from copy import deepcopy

jax_key_order = [ 'conv1.weight', #sshape= torch.Size([64, 3, 7, 7])
  'bn1.weight', #shape= torch.Size([64])
  'bn1.bias', #shape= torch.Size([64])
  'layer1.0.conv1.weight', #shape= torch.Size([64, 64, 3, 3])
  'layer1.0.bn1.weight', #shape= torch.Size([64])
  'layer1.0.bn1.bias', #shape= torch.Size([64])
  'layer1.0.conv2.weight', #shape= torch.Size([64, 64, 3, 3])
  'layer1.0.bn2.weight', #shape= torch.Size([64])
  'layer1.0.bn2.bias', #shape= torch.Size([64])
  'layer1.1.conv1.weight', #shape= torch.Size([64, 64, 3, 3])
  'layer1.1.bn1.weight', #shape= torch.Size([64])
  'layer1.1.bn1.bias', #shape= torch.Size([64])
  'layer1.1.conv2.weight', #shape= torch.Size([64, 64, 3, 3])
  'layer1.1.bn2.weight', #shape= torch.Size([64])
  'layer1.1.bn2.bias', #shape=torch.Size([64])
  'layer2.0.downsample.0.weight', #shape= torch.Size([128, 64, 1, 1])
  'layer2.0.downsample.1.weight', #shape=torch.Size([128])
  'layer2.0.downsample.1.bias', #shape= torch.Size([128])
  'layer2.0.conv1.weight', #shape= torch.Size([128, 64, 3, 3])
  'layer2.0.bn1.weight', #shape= torch.Size([128])
  'layer2.0.bn1.bias', #shape= torch.Size([128])
  'layer2.0.conv2.weight', #shape= torch.Size([128, 128, 3, 3])
  'layer2.0.bn2.weight', #shape= torch.Size([128])
  'layer2.0.bn2.bias', #shape= torch.Size([128])
  'layer2.1.conv1.weight', #shape= torch.Size([128, 128, 3, 3])
  'layer2.1.bn1.weight', #shape= torch.Size([128])
  'layer2.1.bn1.bias', #shape= torch.Size([128])
  'layer2.1.conv2.weight', #shape= torch.Size([128, 128, 3, 3])
  'layer2.1.bn2.weight', #shape= torch.Size([128])
  'layer2.1.bn2.bias', #shape= torch.Size([128])
  'layer3.0.downsample.0.weight', #shape= torch.Size([256, 128, 1, 1])
  'layer3.0.downsample.1.weight', #shape= torch.Size([256])
  'layer3.0.downsample.1.bias', #shape= torch.Size([256])
  'layer3.0.conv1.weight', #shape= torch.Size([256, 128, 3, 3])
  'layer3.0.bn1.weight', #shape= torch.Size([256])
  'layer3.0.bn1.bias', #shape= torch.Size([256])
  'layer3.0.conv2.weight', #shape= torch.Size([256, 256, 3, 3])
  'layer3.0.bn2.weight', #shape= torch.Size([256])
  'layer3.0.bn2.bias', #shape= torch.Size([256])
  'layer3.1.conv1.weight', #shape= torch.Size([256, 256, 3, 3])
  'layer3.1.bn1.weight', #shape= torch.Size([256])
  'layer3.1.bn1.bias', #shape= torch.Size([256])
  'layer3.1.conv2.weight', #shape= torch.Size([256, 256, 3, 3])
  'layer3.1.bn2.weight', #shape= torch.Size([256])
  'layer3.1.bn2.bias', #shape= torch.Size([256])
  'layer4.0.downsample.0.weight', #shape= torch.Size([512, 256, 1, 1])
  'layer4.0.downsample.1.weight', #shape= torch.Size([512])
  'layer4.0.downsample.1.bias', #shape= torch.Size([512])
  'layer4.0.conv1.weight', #shape= torch.Size([512, 256, 3, 3])
  'layer4.0.bn1.weight', #shape= torch.Size([512])
  'layer4.0.bn1.bias', #shape= torch.Size([512])
  'layer4.0.conv2.weight', #shape= torch.Size([512, 512, 3, 3])
  'layer4.0.bn2.weight', #shape= torch.Size([512]) 
  'layer4.0.bn2.bias', #shape= torch.Size([512])
  'layer4.1.conv1.weight', #shape= torch.Size([512, 512, 3, 3])
  'layer4.1.bn1.weight', #shape= torch.Size([512])
  'layer4.1.bn1.bias', #shape= torch.Size([512])
  'layer4.1.conv2.weight', #shape= torch.Size([512, 512, 3, 3])
  'layer4.1.bn2.weight', #shape= torch.Size([512])
  'layer4.1.bn2.bias', #shape= torch.Size([512])
  'fc.weight', #shape= torch.Size([1000, 512])
  'fc.bias' #shape= torch.Size([1000])
  ]

def transfer_params_to_jax( theta, pytorch_model ):
        
    dict_model = dict(pytorch_model.named_parameters())
    jax_weights = [ jax.numpy.array( dict_model[key].detach() ) for key in jax_key_order ]

    jp = []
    length = len(jax_weights)
    cnt = 0
    for jax_params in jax_weights:
        cnt += 1
        if len(jax_params.shape) == 4:
            jax_params = jax_params.transpose( [3,2,1,0] )
        elif len( jax_params.shape ) == 2:
            jax_params = jax_params.transpose( [1,0]  )
        elif len(jax_params.shape) == 1 and cnt != length:
            jax_params = jax_params.reshape( 1,1,1,jax_params.shape[0] )
        jp.append( jax_params )

    jp_gen = iter( jp )#create iterator
    
    #traverse theta and copy the values from jp_gen
    for i in theta:
        for j in theta[i].keys():
            next_params = next(jp_gen,'End')
            if next_params == 'End':
                return theta
            if theta[i][j].shape != next_params.shape:
                print( theta[i][j].shape, next_params.shape, "[Error shapes do not match!]" )
                exit(0)
            else:
                theta[i][j] = deepcopy( next_params )
                #print( theta[i][j].shape, next_params.shape )
            
    return theta