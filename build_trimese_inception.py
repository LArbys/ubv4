import os,sys
import layer_tools as lt
import caffe
from caffe import params as P
from caffe import layers as L

augment_data = True
use_batch_norm = True
use_dropout = False

def root_data_layer( net, batch_size, config, filler_name ):
    net.data, net.label = L.ROOTData( ntop=2, batch_size=batch_size, filler_config=config, filler_name=filler_name )
    return [net.data],net.label

def root_data_layer_trimese( net, batch_size, config, filler_name, slice_points ):
    data, label = root_data_layer( net, batch_size, config, filler_name )
    slices = L.Slice(data[0], ntop=3, name="data_trimese", slice_param=dict(axis=1, slice_point=slice_points))
    return slices, label

def stem( corename, net, data_top, addbatchnorm=True, train=True ):
    conv1 = lt.convolution_layer( net, data_top, "stem_conv1_%s"%(corename), "stem_conv1_%s"%(corename), 
                                  32, 2, 3, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    conv2 = lt.convolution_layer( net, conv1,    "stem_conv2_%s"%(corename), "stem_conv2_%s"%(corename), 
                                  32, 2, 3, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    conv3 = lt.convolution_layer( net, conv2,    "stem_conv3_%s"%(corename), "stem_conv3_%s"%(corename), 
                                  32, 2, 3, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    mp    = lt.pool_layer( net, conv3, "stem_mp1_%s"%(corename), 3, 2 )
    conv4 = lt.convolution_layer( net, conv3, "stem_conv4_%s"%(corename), "stem_conv4_%s"%(corename),
                                  32, 2, 3, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    ls    = [mp,conv4]
    cat   = lt.concat_layer( net, "stem_concat_%s"%(corename), *ls )
    return cat

def buildnet( processcfg, batch_size, height, width, nchannels, user_batch_norm, net_type="train"):
    net = caffe.NetSpec()

    train = False
    if net_type=="train":
        train = True

    data_layers, label = root_data_layer_trimese( net, batch_size, processcfg, net_type, [1,2] )
    stems = []
    for n,data_layer in enumerate(data_layers):
        stems.append( stem( "plane%d"%(n), net, data_layer, user_batch_norm, train ) )

    concat = lt.concat_layer( net, "mergeplanes", *stems )

    # # First conv  layer
    # branch_ends = []
    # for n,layer in enumerate(data_layers):
    #     conv1 = lt.convolution_layer( net, layer, "plane%d_conv1"%(n), "tri_conv1_plane%d"%(n), 64, 2, 5, 3, 0.05, addbatchnorm=True, train=train )
    #     pool1 = lt.pool_layer( net, conv1, "plane%d_pool1"%(n), 3, 1 )

    #     conv2 = lt.convolution_layer( net, pool1, "plane%d_conv2"%(n), "tri_conv2_plane%d"%(n), 64, 2, 3, 3, 0.05, addbatchnorm=True, train=train )
        
    #     conv3 = lt.convolution_layer( net, conv2, "plane%d_conv3"%(n), "tri_conv3_plane%d"%(n), 64, 2, 3, 3, 0.05, addbatchnorm=True, train=train )

    #     pool3 = lt.pool_layer( net, conv3, "plane%d_pool3"%(n), 3, 1 )

    #     branch_ends.append( pool3 )
        
    # concat = lt.concat_layer( net, "mergeplanes", *branch_ends )


    # resnet1  = lt.resnet_module( net, concat,  "resnet1", 64*3, 3, 1, 1,8,32, use_batch_norm, train)
    # resnet2  = lt.resnet_module( net, resnet1, "resnet2", 32, 3, 1, 1,8,32, use_batch_norm, train)
    # resnet3  = lt.resnet_module( net, resnet2, "resnet3", 32, 3, 1, 1,16,64, use_batch_norm, train)
    
    # resnet4  = lt.resnet_module( net, resnet3, "resnet4", 64, 3, 1, 1,16,64, use_batch_norm, train)
    # resnet5  = lt.resnet_module( net, resnet4, "resnet5", 64, 3, 1, 1,16,64, use_batch_norm, train)
    # resnet6  = lt.resnet_module( net, resnet5, "resnet6", 64, 3, 1, 1,32,128, use_batch_norm, train)

    # resnet7  = lt.resnet_module( net, resnet6, "resnet7", 128, 3, 1, 1, 32,128, use_batch_norm, train)
    # resnet8  = lt.resnet_module( net, resnet7, "resnet8", 128, 3, 1, 1, 32,128, use_batch_norm, train)
    # resnet9  = lt.resnet_module( net, resnet8, "resnet9", 128, 3, 1, 1, 64,256, use_batch_norm, train)
        
    # net.lastpool = lt.pool_layer( net, resnet9, "lastpool", 5, 1, P.Pooling.AVE )
    # lastpool_layer = net.lastpool
    
    # if use_dropout:
    #     net.lastpool_dropout = L.Dropout(net.lastpool,
    #                                      in_place=True,
    #                                      dropout_param=dict(dropout_ratio=0.5))
    #     lastpool_layer = net.lastpool_dropout
    
    # fc1 = lt.final_fully_connect( net, lastpool_layer, nclasses=512 )
    # fc2 = lt.final_fully_connect( net, fc1, nclasses=4096 )
    fc2 = lt.final_fully_connect( net, concat, nclasses=2 )
    
    if train:
        net.loss = L.SoftmaxWithLoss(fc2, net.label )
        net.acc = L.Accuracy(fc2,net.label)
    else:
        net.probt = L.Softmax( fc2 )
        net.acc = L.Accuracy(fc2,net.label)

    return net

def append_rootdata_layer( prototxt, imin, imax, flat_mean ):
    fin = open(prototxt,'r')
    fout = open( prototxt.replace(".prototxt","_rootdata.prototxt"), 'w' )
    lines = fin.readlines()
    lout = []
    found_end_of_data = False
    mean_file = ""
    flist = ""
    for l in lines:
        #l = l.strip()
        if found_end_of_data:
            lout.append(l)
        if l=="}\n":
            found_end_of_data = True
        n = len(l.strip().split(":"))
        if n>=2 and l.strip().split(":")[0]=="mean_file":
            mean_file = l.strip().split(":")[1].strip()
        if n>=2 and l.strip().split(":")[0]=="source":
            flist = l.strip().split(":")[1].strip()
        if n>=2 and l.strip().split(":")[0]=="batch_size":
            batch_size = int(l.strip().split(":")[1].strip())
    print mean_file,flist,batch_size

    rootlayer = """
layer {
  name: "data"
  type: "ROOTData"
  top: "data"
  top: "label"

  root_data_param {
    source: %s
    mean: %s
    mean_producer: "mean"
    image_producer: "tpc_hires_crop"
    roi_producer: "tpc_hires_crop"
    nentries: %d
    batch_size: %d
    imin: \"%s\"
    imax: \"%s\"
    flat_mean: \"%s\"
    random_adc_scale_mean: 1.0
    random_adc_scale_sigma: -1.0
    random_col_pad: 0
    random_row_pad: 0
  }    
}
""" % (flist,mean_file,batch_size,batch_size,imin,imax,flat_mean)

    print >>fout,rootlayer

    for l in lout:
        print>>fout,l,
    fin.close()
    fout.close()

if __name__ == "__main__":
    
    train_cfg = "train_filler.cfg"
    test_cfg = "test_filler.cfg"
    use_batch_norm = True

    train_net   = buildnet( train_cfg, 10, 756, 864, 3, use_batch_norm, net_type="train"  )
    test_net    = buildnet( test_cfg,  1, 756, 864, 3, use_batch_norm, net_type="test"  )

    #deploy_net  = buildnet( testdb, test_mean, 1, 768, 768, 3, net_type="deploy"  )

    trainout  = open('ub_trimese_inceptionv4_train.prototxt','w')
    print >> trainout, "name:\"ubv4_train\""
    print >> trainout, train_net.to_proto()
    trainout.close()
    
    testout   = open('ub_trimese_inceptionv4_test.prototxt','w')
    print >> testout, "name:\"ubv4_test\""
    print >> testout, test_net.to_proto()
    testout.close()

    #deployout = open('ub_trimese_resnet_deploy.prototxt','w')
    
    #print >> deployout, deploy_net.to_proto()
    #testout.close()

    #deployout.close()

    #append_rootdata_layer( 'ub_trimese_resnet_train.prototxt', imin, imax, flat_mean )
    #append_rootdata_layer( 'ub_trimese_resnet_test.prototxt',  imin, imax, flat_mean )

    #os.system("rm ub_trimese_resnet_train.prototxt")
    #os.system("rm ub_trimese_resnet_test.prototxt")



