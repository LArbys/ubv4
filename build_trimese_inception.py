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

    # split 7s
    # use same padding up to 3x3
    conv_5a = lt.convolution_layer( net, cat,     "stem_conv5a_%s"%(corename), "stem_conv5a_%s"%(corename),
                                    64, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    conv_5b = lt.convolution_layer( net, conv_5a, "stem_conv5b_%s"%(corename), "stem_conv5b_%s"%(corename),
                                    64, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3 )
    conv_5c = lt.convolution_layer( net, conv_5b, "stem_conv5c_%s"%(corename), "stem_conv5c_%s"%(corename),
                                    64, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0 )
    conv_5d = lt.convolution_layer( net, conv_5c, "stem_conv5d_%s"%(corename), "stem_conv5d_%s"%(corename),
                                    64, 1, 3, 1, 0.0, addbatchnorm=addbatchnorm, train=train )

    # split 3
    conv_6a = lt.convolution_layer( net, cat, "stem_conv6a_%s"%(corename), "stem_conv6a_%s"%(corename),
                                    96, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    conv_6b = lt.convolution_layer( net, conv_6a, "stem_conv6b_%s"%(corename), "stem_conv6b_%s"%(corename),
                                    96, 1, 3, 1, 0.0, addbatchnorm=addbatchnorm, train=train )
    ls2  = [conv_5d, conv_6b]
    cat2 = lt.concat_layer( net, "stem_concat2_%s"%(corename), *ls2 )

    # split 2
    conv7  = lt.convolution_layer( net, cat2, "stem_conv7_%s"%(corename), "stem_conv7_%s"%(corename),
                                   192, 1, 3, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    mp7    = lt.pool_layer( net, cat2, "stem_mp7_%s"%(corename), 3, 1 )

    ls3 = [ conv7, mp7 ]
    cat3 = lt.concat_layer( net, "stem_concat3_%s"%(corename), *ls3 )

    nout = 352

    return cat3, nout

def inceptionA( net, corename, bot, ninputs, noutput, nbottleneck, addbatchnorm=True, train=True ):
    name = corename+"_IA"
    if ninputs!=noutput:
        bypass_conv = L.Convolution( bot,
                                     kernel_size=1,
                                     stride=1,
                                     num_output=noutput,
                                     pad=0,
                                     bias_term=False,
                                     weight_filler=dict(type="msra") )
        if addbatchnorm:
            if train:
                bypass_bn = L.BatchNorm(bypass_conv,in_place=True,batch_norm_param=dict(use_global_stats=False),
                                        param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])
            else:
                bypass_bn = L.BatchNorm(bypass_conv,in_place=True,batch_norm_param=dict(use_global_stats=True))
            bypass_scale = L.Scale(bypass_bn,in_place=True,scale_param=dict(bias_term=True))
            net.__setattr__(name+"_bypass",bypass_conv)
            net.__setattr__(name+"_bypass_bn",bypass_bn)
            net.__setattr__(name+"_bypass_scale",bypass_scale)
        else:
            net.__setattr__(name+"_bypass",bypass_conv)
        bypass_layer = bypass_conv
    else:
        bypass_layer  = bot

    
    conva1 = lt.convolution_layer( net, bot, name+"_conva", name+"_conva",
                                   nbottleneck, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train )

    convb1 = lt.convolution_layer( net, bot, name+"_convb1", name+"_convb1",
                                   nbottleneck, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    convb2 = lt.convolution_layer( net, convb1, name+"_convb2", name+"_convb2",
                                   nbottleneck, 1, 3, 1, 0.0, addbatchnorm=addbatchnorm, train=train )

    convc1 = lt.convolution_layer( net, bot, name+"_convc1", name+"_convc1",
                                   nbottleneck, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    convc2 = lt.convolution_layer( net, convc1, name+"_convc2", name+"_convc2",
                                   nbottleneck, 1, 3, 1, 0.0, addbatchnorm=addbatchnorm, train=train )
    convc3 = lt.convolution_layer( net, convc2, name+"_convc3", name+"_convc3",
                                   nbottleneck, 1, 3, 1, 0.0, addbatchnorm=addbatchnorm, train=train )

    ls = [conva1,convb2,convc3]
    cat    = lt.concat_layer( net, name+"_concat", *ls )

    convd = lt.convolution_layer( net, cat, name+"_convd",name+"_convd",
                                  noutput, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    
    ex_last_layer = convd
    
    # Eltwise
    elt_layer = L.Eltwise(bypass_layer,ex_last_layer, eltwise_param=dict(operation=P.Eltwise.SUM))
    elt_relu  = L.ReLU( elt_layer,in_place=True)
    net.__setattr__(name+"_eltwise",elt_layer)
    net.__setattr__(name+"_eltwise_relu",elt_relu)

    return elt_relu

def reductionA( net, corename, bot, noutN, noutK, noutL, noutM, addbatchnorm=True, train=True ):
    mpa    = lt.pool_layer( net, bot, "reducA_mpA_%s"%(corename), 3, 2, pad_w=1 )
    
    convb  = lt.convolution_layer( net, bot, "reducA_convb_%s"%(corename), "reducA_convb_%s"%(corename),
                                   noutN, 2, 3, 1, 0.0, addbatchnorm=addbatchnorm, train=train, kernel_w=3, kernel_h=3, pad_w=1, pad_h=1 )

    convc1 = lt.convolution_layer( net, bot, "reducA_convc1_%s"%(corename), "reducA_convc1_%s"%(corename),
                                   noutK, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    convc2 = lt.convolution_layer( net, convc1, "reducA_convc2_%s"%(corename), "reducA_convc2_%s"%(corename),
                                   noutL, 1, 3, 1, 0.0, addbatchnorm=addbatchnorm, train=train )
    convc3 = lt.convolution_layer( net, convc2, "reducA_convc3_%s"%(corename), "reducA_convc3_%s"%(corename),
                                   noutM, 2, 3, 1, 0.0, addbatchnorm=addbatchnorm, train=train, kernel_w=3, kernel_h=3, pad_w=1, pad_h=1 )

    ls  = [mpa,convb,convc3]
    cat = lt.concat_layer( net, "reducA_concat_%s"%(corename), *ls )

    return cat

def inceptionB( net, corename, bot, addbatchnorm=True, train=True ):
    apa   = lt.pool_layer( net, bot, "IB_avepool_%s"%(corename), 3, 1, pooltype=P.Pooling.AVE, pad_w=1, pad_h=1 )
    conva = lt.convolution_layer( net, apa, "IB_conva_%s"%(corename), "IB_conva_%s"%(corename),
                                  128, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm, train=train )
    
    convb = lt.convolution_layer( net, bot, "IB_convb_%s"%(corename), "IB_convb_%s"%(corename),
                                  384, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm,train=train )
    
    convc1 = lt.convolution_layer( net, bot, "IB_convc1_%s"%(corename), "IB_convc1_%s"%(corename),
                                   192, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm,train=train )
    convc2 = lt.convolution_layer( net, convc1, "IB_convc2_%s"%(corename), "IB_convc2_%s"%(corename),
                                   224, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm,train=train, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3 )
    convc3 = lt.convolution_layer( net, convc2, "IB_convc3_%s"%(corename), "IB_convc3_%s"%(corename),
                                   256, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm,train=train, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0 )

    convd1 = lt.convolution_layer( net, bot, "IB_convd1_%s"%(corename), "IB_convd1_%s"%(corename),
                                   192, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm,train=train )
    convd2 = lt.convolution_layer( net, convd1, "IB_convd2_%s"%(corename),"IB_convd2_%s"%(corename),
                                   192, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm,train=train, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3 )
    convd3 = lt.convolution_layer( net, convd2, "IB_convd3_%s"%(corename),"IB_convd3_%s"%(corename),
                                   224, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm,train=train, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0 )
    convd4 = lt.convolution_layer( net, convd3, "IB_convd4_%s"%(corename),"IB_convd4_%s"%(corename),
                                   224, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm,train=train, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3 )
    convd5 = lt.convolution_layer( net, convd4, "IB_convd5_%s"%(corename),"IB_convd5_%s"%(corename),
                                   256, 1, 1, 0, 0.0, addbatchnorm=addbatchnorm,train=train, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0 )

    ls = [ conva, convb, convc3, convd5 ]
    cat = lt.concat_layer( net, "inductB_concat_%s"%(corename), *ls )
    
    return cat


def buildnet( processcfg, batch_size, height, width, nchannels, user_batch_norm, net_type="train"):
    net = caffe.NetSpec()

    train = False
    if net_type=="train":
        train = True

    data_layers, label = root_data_layer_trimese( net, batch_size, processcfg, net_type, [1,2] )
    stems = []
    for n,data_layer in enumerate(data_layers):
        outstem,nout = stem( "plane%d"%(n), net, data_layer, addbatchnorm=True, train=train )
        ia1     = inceptionA( net, "ia1_plane%d"%(n), outstem, nout, 256, 32, addbatchnorm=False, train=train )
        ia2     = inceptionA( net, "ia2_plane%d"%(n),     ia1, 256, 256, 32, addbatchnorm=False, train=train )
        ia3     = inceptionA( net, "ia3_plane%d"%(n),     ia2, 256, 256, 32, addbatchnorm=False, train=train )
        reda    = reductionA( net, "plane%d"%(n),       ia3, 32, 32, 32, 32, addbatchnorm=False, train=train )
        ib1     = inceptionB( net, "ib1_plane%d"%(n), reda, addbatchnorm=False, train=train )
        ib2     = inceptionB( net, "ib2_plane%d"%(n),  ib1, addbatchnorm=False, train=train )
        ib3     = inceptionB( net, "ib3_plane%d"%(n),  ib2, addbatchnorm=False, train=train )
        stems.append( reda  ) # no batch norm for stem. too many parameters!


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



