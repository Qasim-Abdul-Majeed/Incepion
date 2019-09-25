import tensorflow as tf


def get_variable(name, shape, train):
    return tf.get_variable(name,
                           shape = shape,
                           dtype = tf.float32,
                           trainable = train
                          )
def conv_layer(inputs,
               lay_name = 'conv', #Name of a layer.
               filt_shape = [3, 3],
               ch_in = 3,
               ch_out = 1,
               train = False,
               bias = False,
               padd = 'SAME',
               stride = [1, 1, 1, 1]):
    filt = get_variable(name = 'f_' + lay_name,
                        shape = [filt_shape[0], filt_shape[1], ch_in,  ch_out],
                        train = train)
    conv = tf.nn.conv2d(inputs, filt, strides = stride, padding =  padd, name = lay_name)
    
    if bias:
        bias_val = get_variable('b_' + lay_name,
                                ch_out,
                                train)
        return tf.nn.relu(tf.nn.bias_add(conv, bias_val))
    return tf.nn.relu(conv)
	
#Place holders for network input.
batch_size = 8
pred = 14 #Number of predictions our model have to predict per image.
x = tf.placeholder(shape = (batch_size, 299, 299, 3), name = 'inputs', dtype = tf.float32)
y =  tf.placeholder(shape = (batch_size, pred), name = 'labels', dtype = tf.float32)


#Define a model...
conv_1 = conv_layer(x,
                    'conv_1',
                    filt_shape = [3, 3],
                    ch_in = x.shape[-1],
                    ch_out = 32,
                    train = False,
                    padd = 'VALID',
                   stride = [1, 2, 2, 1])
print(conv_1)

conv_2 = conv_layer(conv_1,
                   'conv_2',
                   filt_shape = [3, 3],
                   ch_in = conv_1.shape[-1],
                   ch_out = 32,
                   train = False,
                   padd = 'VALID')
print(conv_2)

conv_3 = conv_layer(conv_2,
                   'conv_3',
                   filt_shape = [3, 3],
                   ch_in = conv_2.shape[-1],
                   ch_out = 64,
                   train = False,
                   padd = 'SAME',
                   stride = [1, 1, 1, 1])
print(conv_3)

max_pool_1 = tf.nn.max_pool(conv_3, 
                            ksize = [1, 3, 3, 1],
                            strides = [1, 2, 2, 1],
                            padding = 'VALID',
                            name = 'pool1')
print(max_pool_1)

conv_4 =  conv_layer(max_pool_1,
                     'conv_4',
                     filt_shape = [1, 1],
                     ch_in = max_pool_1.shape[-1],
                     ch_out = 80,
                     train = False,
                     padd = 'VALID',
                     stride = [1, 1, 1, 1])
print(conv_4)
'''
conv_5 =  conv_layer(conv_4,
                     'conv_5',
                     filt_shape = [3,3],
                     ch_in = conv_4.shape[-1],
                     ch_out = 192,
                     train = False,
                     padd = 'VALID',
                     stride = [1, 2, 2, 1]
                    )
'''
#Copy from a keras_inception_v3
conv_5 =  conv_layer(conv_4,
                     'conv_5',
                     filt_shape = [3,3],
                     ch_in = conv_4.shape[-1],
                     ch_out = 192,
                     train = False,
                     padd = 'VALID',
                     stride = [1, 1, 1, 1]
                    )
print(conv_5)

#Now copy flow from its official github...
max_pool_2 =  tf.nn.max_pool(conv_5,
                                  ksize = [1, 3, 3, 1],
                                  strides = [1, 2, 2, 1],
                                  padding = 'VALID',
                                  name = 'max_pool_5a_3x3'
                                 )
print(max_pool_2)

#Now building Inception Blocks...
#Mixed_5b
branch_0 = conv_layer(max_pool_2,
                      'conv_6_b0',
                      filt_shape = [1, 1],
                      ch_in = max_pool_2.shape[-1],
                      ch_out = 64,
                      padd = 'VALID',
                      stride = [1, 1, 1, 1]
                      )
print(branch_0)

branch_1_a = conv_layer(max_pool_2,
                      'conv_6_b1_a',
                      filt_shape = [1, 1],
                      ch_in = max_pool_2.shape[-1],
                      ch_out = 48,
                      padd = 'VALID',
                      stride = [1, 1, 1, 1]
                     )
print(branch_1_a)
branch_1_b = conv_layer(branch_1_a,
                        'conv_6_b1_b',
                        filt_shape = [5, 5],
                        ch_in = branch_1_a.shape[-1],
                        ch_out = 64,
                        padd = 'SAME'
                       )
print(branch_1_b)

branch_2_a = conv_layer(max_pool_2,
                       'conv_6_b2_a',
                       filt_shape = [1, 1],
                       ch_in = max_pool_2.shape[-1],
                       ch_out = 64
                      )
print(branch_2_a)
branch_2_b = conv_layer(branch_2_a,
                        'conv_6_b2_b',
                        filt_shape = [3, 3],
                        ch_in = branch_2_a.shape[-1],
                        ch_out = 96
                       )
print(branch_2_b)
branch_2_c = conv_layer(branch_2_b,
                        'conv_6_b2_c',
                        filt_shape = [3, 3],
                        ch_in = branch_2_b.shape[-1],
                        ch_out = 96
                       )
print(branch_2_c)

avg_pool_b3_a = tf.nn.avg_pool(max_pool_2,
                            ksize = [1, 3, 3, 1],
                            strides = [1, 1, 1, 1],
                            padding = 'SAME',
                            name = 'avg_pool_b3_a',
                        )
print(avg_pool_b3_a)
branch_3_b = conv_layer(avg_pool_b3_a,
                        'conv_6_b3_b',
                        filt_shape = [1, 1],
                        ch_in = avg_pool_b3_a.shape[-1],
                        ch_out = 32
                       )
print(branch_3_b)

conv_6 = tf.concat(axis = 3,
                   values = [branch_0, branch_1_b, branch_2_c, branch_3_b])
print(conv_6)


#Now Mixed_5c
mixed_5c_b0 = conv_layer(conv_6,
                         'mixed_5c_b0',
                         filt_shape = [1, 1],
                         ch_in = conv_6.shape[-1],
                         ch_out = 64
                        )
print(mixed_5c_b0)

mixed_5c_b1_a = conv_layer(conv_6,
                         'mixed_5c_b1_a',
                          filt_shape = [1, 1],
                          ch_in = conv_6.shape[-1],
                          ch_out = 48
                        )
print(mixed_5c_b1_a)
mixed_5c_b1_b = conv_layer(mixed_5c_b1_a,
                           'mixed_5c_b1_b',
                           filt_shape = [5, 5],
                           ch_in = mixed_5c_b1_a.shape[-1],
                           ch_out = 64
                          )
print(mixed_5c_b1_b)

mixed_5c_b2_a = conv_layer(conv_6,
                           'mixed_5c_b2_a',
                           filt_shape = [1, 1],
                           ch_in = conv_6.shape[-1],
                           ch_out = 64
                          )
print(mixed_5c_b2_a)
mixed_5c_b2_b = conv_layer(mixed_5c_b2_a,
                           'mixed_5c_b2_b',
                           filt_shape =  [3, 3],
                           ch_in =  mixed_5c_b2_a.shape[-1],
                           ch_out = 96
                          )
print(mixed_5c_b2_b)
mixed_5c_b2_c = conv_layer(mixed_5c_b2_b,
                           'mixed_5c_b2_c',
                           filt_shape = [3, 3],
                           ch_in = mixed_5c_b2_b.shape[-1],
                           ch_out = 96
                          )
print(mixed_5c_b2_c)


mixed_5c_b3_avg = tf.nn.avg_pool(conv_6,
                                 ksize =  [1, 3, 3, 1],
                                 strides = [1, 1, 1, 1],
                                 padding =  'SAME'
                                )
print(mixed_5c_b3_avg)
mixed_5c_b3_b = conv_layer(mixed_5c_b3_avg,
                           'mixed_5c_b3_b',
                           filt_shape = [1, 1],
                           ch_in = mixed_5c_b3_avg.shape[-1],
                           ch_out = 64
                          )
print(mixed_5c_b3_b)

mixed_5c = tf.concat(axis  = 3,
                     values = [mixed_5c_b0, mixed_5c_b1_b, mixed_5c_b2_c, mixed_5c_b3_b])
print(mixed_5c)

#Now Mixed_5d
mixed_5d_b0 = conv_layer(mixed_5c,
                         'mixed_5d_b0',
                         filt_shape = [1, 1],
                         ch_in = mixed_5c.shape[-1],
                         ch_out = 64
                        )
print(mixed_5d_b0)

mixed_5d_b1_a = conv_layer(mixed_5c,
                         'mixed_5d_b1_a',
                          filt_shape = [1, 1],
                          ch_in = mixed_5c.shape[-1],
                          ch_out = 48
                        )
print(mixed_5d_b1_a)
mixed_5d_b1_b = conv_layer(mixed_5d_b1_a,
                           'mixed_5d_b1_b',
                           filt_shape = [5, 5],
                           ch_in = mixed_5d_b1_a.shape[-1],
                           ch_out = 64
                          )
print(mixed_5d_b1_b)

mixed_5d_b2_a = conv_layer(mixed_5c,
                           'mixed_5d_b2_a',
                           filt_shape = [1, 1],
                           ch_in = mixed_5c.shape[-1],
                           ch_out = 64
                          )
print(mixed_5d_b2_a)
mixed_5d_b2_b = conv_layer(mixed_5d_b2_a,
                           'mixed_5d_b2_b',
                           filt_shape =  [3, 3],
                           ch_in =  mixed_5d_b2_a.shape[-1],
                           ch_out = 96
                          )
print(mixed_5d_b2_b)
mixed_5d_b2_c = conv_layer(mixed_5d_b2_b,
                           'mixed_5d_b2_c',
                           filt_shape = [3, 3],
                           ch_in = mixed_5d_b2_b.shape[-1],
                           ch_out = 96
                          )
print(mixed_5d_b2_c)


mixed_5d_b3_avg = tf.nn.avg_pool(mixed_5c,
                                 ksize =  [1, 3, 3, 1],
                                 strides = [1, 1, 1, 1],
                                 padding =  'SAME'
                                )
print(mixed_5d_b3_avg)
mixed_5d_b3_b = conv_layer(mixed_5d_b3_avg,
                           'mixed_5d_b3_b',
                           filt_shape = [1, 1],
                           ch_in = mixed_5d_b3_avg.shape[-1],
                           ch_out = 64
                          )
print(mixed_5d_b3_b)

mixed_5d = tf.concat(axis  = 3,
                     values = [mixed_5d_b0, mixed_5d_b1_b, mixed_5d_b2_c, mixed_5d_b3_b])
print(mixed_5d)


#Now Mixed_6a
mixed_6a_b0 = conv_layer(mixed_5d,
                       'mixed_6a_b0',
                       filt_shape = [3, 3],
                       ch_in = mixed_5d.shape[-1],
                       ch_out = 384,
                       padd =  'VALID',
                       stride = [1, 2, 2, 1]
                      )
print(mixed_6a_b0)

mixed_6a_b1_a = conv_layer(mixed_5d,
                       'mixed_6a_b1_a',
                       filt_shape = [1, 1],
                       ch_in = mixed_5d.shape[-1],
                       ch_out = 64,
                       stride = [1, 1, 1, 1]
                      )
print(mixed_6a_b1_a)
mixed_6a_b1_b = conv_layer(mixed_6a_b1_a,
                         'mixed_6a_b1_b',
                         filt_shape = [3, 3],
                         ch_in = mixed_6a_b1_a.shape[-1],
                         ch_out =  96
                        )
print(mixed_6a_b1_b)
mixed_6a_b1_c = conv_layer(mixed_6a_b1_b,
                         'mixed_6a_b1_c',
                         filt_shape = [3, 3],
                         ch_in = mixed_6a_b1_b.shape[-1],
                         ch_out =  96,
                         padd =  'VALID',
                         stride = [1, 2, 2, 1]
                        )
print(mixed_6a_b1_c)

mixed_6a_b2_max = tf.nn.max_pool(mixed_5d,
                                 ksize = [1, 3, 3, 1],
                                 strides = [1, 2, 2, 1],
                                 padding = 'VALID',
                                 name = 'mixed_6a_b2_max' 
                                )
print(mixed_6a_b2_max)

mixed_6a = tf.concat(axis = 3,
                  values = [mixed_6a_b0, mixed_6a_b1_c, mixed_6a_b2_max])
print(mixed_6a)

#Now Mixed_6b

mixed_6b_b0 = conv_layer(mixed_6a,
                       lay_name = 'mixed_6b_b0',
                       filt_shape = [1, 1],
                       ch_in = mixed_6a.shape[-1],
                       ch_out = 192
                      )
print(mixed_6a_b0)

mixed_6b_b1_a = conv_layer(mixed_6a,
                       'mixed_6b_b1_a',
                        filt_shape = [1, 1],
                        ch_in = mixed_6a.shape[-1],
                        ch_out = 128
                      )
print(mixed_6b_b1_a)
mixed_6b_b1_b = conv_layer(mixed_6b_b1_a,
                         'mixed_6b_b1_b',
                         filt_shape = [1, 7],
                         ch_in = mixed_6b_b1_a.shape[-1],
                         ch_out = 128
                        )
print(mixed_6b_b1_b)
mixed_6b_b1_c = conv_layer(mixed_6b_b1_b,
                         'mixed_6b_b1_c',
                         filt_shape = [7, 1],
                         ch_in = mixed_6b_b1_b.shape[-1],
                         ch_out = 192
                        )
print(mixed_6b_b1_c)

mixed_6b_b2_a = conv_layer(mixed_6a,
                         'mixed_6b_b2_a',
                         filt_shape = [1, 1],
                         ch_in = mixed_6a.shape[-1],
                         ch_out = 128
                        )
print(mixed_6b_b2_a)
mixed_6b_b2_b = conv_layer(mixed_6b_b2_a,
                         'mixed_6b_b2_b',
                         filt_shape = [7, 1],
                         ch_in =  mixed_6b_b2_a.shape[-1],
                         ch_out = 128
                        )
print(mixed_6b_b2_b)
mixed_6b_b2_c = conv_layer(mixed_6b_b2_b,
                         'mixed_6b_b2_c',
                         filt_shape = [1, 7],
                         ch_in = mixed_6b_b2_b.shape[-1],
                         ch_out = 128
                        )
print(mixed_6b_b2_c)
mixed_6b_b2_d = conv_layer(mixed_6b_b2_c,
                         'mixed_6b_b2_d',
                         filt_shape = [7, 1],
                         ch_in = mixed_6b_b2_c.shape[-1],
                         ch_out = 128
                        )
print(mixed_6b_b2_d)
mixed_6b_b2_e = conv_layer(mixed_6b_b2_d,
                         'mixed_6b_b2_e',
                         filt_shape = [1, 7],
                         ch_in = mixed_6b_b2_d.shape[-1],
                         ch_out = 192
                        )
print(mixed_6b_b2_e)

mixed_6b_b3_avg = tf.nn.avg_pool(mixed_6a,
                               ksize = [1, 3, 3, 1],
                               strides = [1, 1, 1, 1],
                               padding = 'SAME'
                              )
print(mixed_6b_b3_avg)
mixed_6b_b3_b = conv_layer(mixed_6b_b3_avg,
                         'mixed_6b_b3_b',
                         filt_shape = [1, 1],
                         ch_in = mixed_6b_b3_avg.shape[-1],
                         ch_out = 192
                        )
print(mixed_6b_b3_b)

mixed_6b = tf.concat(axis = 3, 
                   values = [mixed_6b_b0, mixed_6b_b1_c, mixed_6b_b2_e, mixed_6b_b3_b])

print(mixed_6b)

#Now Mixed_6c

mixed_6c_b0 = conv_layer(mixed_6b,
                       lay_name = 'mixed_6c_b0',
                       filt_shape = [1, 1],
                       ch_in = mixed_6b.shape[-1],
                       ch_out = 192
                      )
print(mixed_6c_b0)

mixed_6c_b1_a = conv_layer(mixed_6b,
                       'mixed_6c_b1_a',
                        filt_shape = [1, 1],
                        ch_in = mixed_6b.shape[-1],
                        ch_out = 160
                      )
print(mixed_6c_b1_a)
mixed_6c_b1_b = conv_layer(mixed_6c_b1_a,
                         'mixed_6c_b1_b',
                         filt_shape = [1, 7],
                         ch_in = mixed_6c_b1_a.shape[-1],
                         ch_out = 160
                        )
print(mixed_6c_b1_b)
mixed_6c_b1_c = conv_layer(mixed_6c_b1_b,
                         'mixed_6c_b1_c',
                         filt_shape = [7, 1],
                         ch_in = mixed_6c_b1_b.shape[-1],
                         ch_out = 192
                        )
print(mixed_6c_b1_c)

mixed_6c_b2_a = conv_layer(mixed_6b,
                         'mixed_6c_b2_a',
                         filt_shape = [1, 1],
                         ch_in = mixed_6b.shape[-1],
                         ch_out = 160
                        )
print(mixed_6c_b2_a)
mixed_6c_b2_b = conv_layer(mixed_6c_b2_a,
                         'mixed_6c_b2_b',
                         filt_shape = [7, 1],
                         ch_in =  mixed_6c_b2_a.shape[-1],
                         ch_out = 160
                        )
print(mixed_6c_b2_b)
mixed_6c_b2_c = conv_layer(mixed_6c_b2_b,
                         'mixed_6c_b2_c',
                         filt_shape = [1, 7],
                         ch_in = mixed_6c_b2_b.shape[-1],
                         ch_out = 160
                        )
print(mixed_6c_b2_c)
mixed_6c_b2_d = conv_layer(mixed_6c_b2_c,
                         'mixed_6c_b2_d',
                         filt_shape = [7, 1],
                         ch_in = mixed_6c_b2_c.shape[-1],
                         ch_out = 160
                        )
print(mixed_6c_b2_d)
mixed_6c_b2_e = conv_layer(mixed_6c_b2_d,
                         'mixed_6c_b2_e',
                         filt_shape = [1, 7],
                         ch_in = mixed_6c_b2_d.shape[-1],
                         ch_out = 192
                        )
print(mixed_6c_b2_e)

mixed_6c_b3_avg = tf.nn.avg_pool(mixed_6b,
                               ksize = [1, 3, 3, 1],
                               strides = [1, 1, 1, 1],
                               padding = 'SAME'
                              )
print(mixed_6c_b3_avg)
mixed_6c_b3_b = conv_layer(mixed_6c_b3_avg,
                         'mixed_6c_b3_b',
                         filt_shape = [1, 1],
                         ch_in = mixed_6c_b3_avg.shape[-1],
                         ch_out = 192
                        )
print(mixed_6c_b3_b)

mixed_6c = tf.concat(axis = 3, 
                   values = [mixed_6c_b0, mixed_6c_b1_c, mixed_6c_b2_e, mixed_6c_b3_b])

print(mixed_6c)


#Now Mixed_6d

mixed_6d_b0 = conv_layer(mixed_6c,
                       lay_name = 'mixed_6d_b0',
                       filt_shape = [1, 1],
                       ch_in = mixed_6c.shape[-1],
                       ch_out = 192
                      )
print(mixed_6d_b0)

mixed_6d_b1_a = conv_layer(mixed_6c,
                       'mixed_6d_b1_a',
                        filt_shape = [1, 1],
                        ch_in = mixed_6c.shape[-1],
                        ch_out = 160
                      )
print(mixed_6d_b1_a)
mixed_6d_b1_b = conv_layer(mixed_6d_b1_a,
                         'mixed_6d_b1_b',
                         filt_shape = [1, 7],
                         ch_in = mixed_6d_b1_a.shape[-1],
                         ch_out = 160
                        )
print(mixed_6d_b1_b)
mixed_6d_b1_c = conv_layer(mixed_6d_b1_b,
                         'mixed_6d_b1_c',
                         filt_shape = [7, 1],
                         ch_in = mixed_6d_b1_b.shape[-1],
                         ch_out = 192
                        )
print(mixed_6d_b1_c)

mixed_6d_b2_a = conv_layer(mixed_6c,
                         'mixed_6d_b2_a',
                         filt_shape = [1, 1],
                         ch_in = mixed_6c.shape[-1],
                         ch_out = 160
                        )
print(mixed_6d_b2_a)
mixed_6d_b2_b = conv_layer(mixed_6d_b2_a,
                         'mixed_6d_b2_b',
                         filt_shape = [7, 1],
                         ch_in =  mixed_6d_b2_a.shape[-1],
                         ch_out = 160
                        )
print(mixed_6d_b2_b)
mixed_6d_b2_c = conv_layer(mixed_6d_b2_b,
                         'mixed_6d_b2_c',
                         filt_shape = [1, 7],
                         ch_in = mixed_6d_b2_b.shape[-1],
                         ch_out = 160
                        )
print(mixed_6d_b2_c)
mixed_6d_b2_d = conv_layer(mixed_6d_b2_c,
                         'mixed_6d_b2_d',
                         filt_shape = [7, 1],
                         ch_in = mixed_6d_b2_c.shape[-1],
                         ch_out = 160
                        )
print(mixed_6d_b2_d)
mixed_6d_b2_e = conv_layer(mixed_6d_b2_d,
                         'mixed_6d_b2_e',
                         filt_shape = [1, 7],
                         ch_in = mixed_6d_b2_d.shape[-1],
                         ch_out = 192
                        )
print(mixed_6d_b2_e)

mixed_6d_b3_avg = tf.nn.avg_pool(mixed_6c,
                               ksize = [1, 3, 3, 1],
                               strides = [1, 1, 1, 1],
                               padding = 'SAME'
                              )
print(mixed_6d_b3_avg)
mixed_6d_b3_b = conv_layer(mixed_6d_b3_avg,
                         'mixed_6d_b3_b',
                         filt_shape = [1, 1],
                         ch_in = mixed_6d_b3_avg.shape[-1],
                         ch_out = 192
                        )
print(mixed_6d_b3_b)

mixed_6d = tf.concat(axis = 3, 
                   values = [mixed_6d_b0, mixed_6d_b1_c, mixed_6d_b2_e, mixed_6d_b3_b])

print(mixed_6d)


#Now Mixed_6e

mixed_6e_b0 = conv_layer(mixed_6d,
                       lay_name = 'mixed_6e_b0',
                       filt_shape = [1, 1],
                       ch_in = mixed_6d.shape[-1],
                       ch_out = 192
                      )
print(mixed_6e_b0)

mixed_6e_b1_a = conv_layer(mixed_6d,
                       'mixed_6e_b1_a',
                        filt_shape = [1, 1],
                        ch_in = mixed_6d.shape[-1],
                        ch_out = 192
                      )
print(mixed_6e_b1_a)
mixed_6e_b1_b = conv_layer(mixed_6e_b1_a,
                         'mixed_6e_b1_b',
                         filt_shape = [1, 7],
                         ch_in = mixed_6e_b1_a.shape[-1],
                         ch_out = 192
                        )
print(mixed_6e_b1_b)
mixed_6e_b1_c = conv_layer(mixed_6e_b1_b,
                         'mixed_6e_b1_c',
                         filt_shape = [7, 1],
                         ch_in = mixed_6e_b1_b.shape[-1],
                         ch_out = 192
                        )
print(mixed_6e_b1_c)

mixed_6e_b2_a = conv_layer(mixed_6d,
                         'mixed_6e_b2_a',
                         filt_shape = [1, 1],
                         ch_in = mixed_6d.shape[-1],
                         ch_out = 192
                        )
print(mixed_6e_b2_a)
mixed_6e_b2_b = conv_layer(mixed_6e_b2_a,
                         'mixed_6e_b2_b',
                         filt_shape = [7, 1],
                         ch_in =  mixed_6e_b2_a.shape[-1],
                         ch_out = 192
                        )
print(mixed_6e_b2_b)
mixed_6e_b2_c = conv_layer(mixed_6e_b2_b,
                         'mixed_6e_b2_c',
                         filt_shape = [1, 7],
                         ch_in = mixed_6e_b2_b.shape[-1],
                         ch_out = 192
                        )
print(mixed_6e_b2_c)
mixed_6e_b2_d = conv_layer(mixed_6e_b2_c,
                         'mixed_6e_b2_d',
                         filt_shape = [7, 1],
                         ch_in = mixed_6e_b2_c.shape[-1],
                         ch_out = 192
                        )
print(mixed_6e_b2_d)
mixed_6e_b2_e = conv_layer(mixed_6e_b2_d,
                         'mixed_6e_b2_e',
                         filt_shape = [1, 7],
                         ch_in = mixed_6e_b2_d.shape[-1],
                         ch_out = 192
                        )
print(mixed_6e_b2_e)

mixed_6e_b3_avg = tf.nn.avg_pool(mixed_6d,
                               ksize = [1, 3, 3, 1],
                               strides = [1, 1, 1, 1],
                               padding = 'SAME'
                              )
print(mixed_6e_b3_avg)
mixed_6e_b3_b = conv_layer(mixed_6e_b3_avg,
                         'mixed_6e_b3_b',
                         filt_shape = [1, 1],
                         ch_in = mixed_6e_b3_avg.shape[-1],
                         ch_out = 192
                        )
print(mixed_6e_b3_b)

mixed_6e = tf.concat(axis = 3, 
                   values = [mixed_6e_b0, mixed_6e_b1_c, mixed_6e_b2_e, mixed_6e_b3_b])

print(mixed_6e)

#Now Mixed_7a

mixed_7a_b0_a = conv_layer(mixed_6e,
                         'mixed_7a_b0_a',
                         filt_shape = [1, 1],
                         ch_in = mixed_6e.shape[-1],
                         ch_out = 192
                        )
print(mixed_7a_b0_a)
mixed_7a_b0_b = conv_layer(mixed_7a_b0_a,
                           'mixed_7a_b0_b',
                           filt_shape = [3, 3],
                           ch_in = mixed_7a_b0_a.shape[-1],
                           ch_out = 320,
                           padd = 'VALID',
                           stride = [1, 2, 2, 1]
                          )
print(mixed_7a_b0_b)

mixed_7a_b1_a = conv_layer(mixed_6e,
                           'mixed_7a_b1_a',
                           filt_shape = [1, 1],
                           ch_in = mixed_6e.shape[-1],
                           ch_out = 192
                          )
print(mixed_7a_b1_a)
mixed_7a_b1_b = conv_layer(mixed_7a_b1_a,
                           'mixed_7a_b1_b',
                           filt_shape = [1, 7],
                           ch_in = mixed_7a_b1_a.shape[-1],
                           ch_out = 192
                          )
print(mixed_7a_b1_b)
mixed_7a_b1_c = conv_layer(mixed_7a_b1_b,
                           'mixed_7a_b1_c',
                           filt_shape = [7, 1],
                           ch_in = mixed_7a_b1_b.shape[-1],
                           ch_out = 192
                          )
print(mixed_7a_b1_c)
mixed_7a_b1_d = conv_layer(mixed_7a_b1_c,
                           'mixed_7a_b1_d',
                           filt_shape = [3, 3],
                           ch_in = mixed_7a_b1_c.shape[-1],
                           ch_out = 192,
                           stride = [1, 2, 2, 1],
                           padd = 'VALID'
                          )
print(mixed_7a_b1_d)

mixed_7a_b2_max = tf.nn.max_pool(mixed_6e,
                             ksize = [1, 3, 3, 1],
                             strides = [1, 2, 2, 1],
                             padding = 'VALID',
                             name = 'mixed_7a_b2_max'
                            )
print(mixed_7a_b2_max)

mixed_7a = tf.concat(axis = 3,
                    values = [mixed_7a_b0_b, mixed_7a_b1_d, mixed_7a_b2_max])
print(mixed_7a)


#Now Mixed_7b

mixed_7b_b0 = conv_layer(mixed_7a,
                           'mixed_7b_b0',
                           filt_shape = [1, 1],
                           ch_in = mixed_7a.shape[-1],
                           ch_out = 320
                          )
print(mixed_7b_b0)

mixed_7b_b1_a = conv_layer(mixed_7a,
                           'mixed_7b_b1_a',
                           filt_shape = [1, 1],
                           ch_in = mixed_7a.shape[-1],
                           ch_out = 384
                          )
print(mixed_7b_b1_a)
mixed_7b_b1_b_1x3 = conv_layer(mixed_7b_b1_a,
                           'mixed_7b_b1_a_1x3',
                           filt_shape = [1, 3],
                           ch_in = mixed_7b_b1_a.shape[-1],
                           ch_out = 384
                          )
print(mixed_7b_b1_b_1x3)
mixed_7b_b1_b_3x1 = conv_layer(mixed_7b_b1_a,
                           'mixed_7b_b1_a_3x1',
                           filt_shape = [3, 1],
                           ch_in = mixed_7b_b1_a.shape[-1],
                           ch_out = 384
                          )
print(mixed_7b_b1_b_3x1)
mixed_7b_b1 = tf.concat(axis = 3,
                       values = [mixed_7b_b1_b_1x3, mixed_7b_b1_b_3x1])
print(mixed_7b_b1)

mixed_7b_b2_a = conv_layer(mixed_7a,
                           'mixed_7b_b2_a',
                           filt_shape = [1, 1],
                           ch_in = mixed_7a.shape[-1],
                           ch_out = 448
                          )
print(mixed_7b_b2_a)
mixed_7b_b2_b = conv_layer(mixed_7b_b2_a,
                           'mixed_7b_b2_b',
                           filt_shape = [3, 3],
                           ch_in = mixed_7b_b2_a.shape[-1],
                           ch_out = 384
                          )
print(mixed_7b_b2_b)

mixed_7b_b2_c_1x3 = conv_layer(mixed_7b_b2_b,
                           'mixed_7b_b2_c_1x3',
                           filt_shape = [1, 3],
                           ch_in = mixed_7b_b2_b.shape[-1],
                           ch_out = 384
                          )
print(mixed_7b_b2_c_1x3)
mixed_7b_b2_c_3x1 = conv_layer(mixed_7b_b2_b,
                           'mixed_7b_b2_c_3x1',
                           filt_shape = [3, 1],
                           ch_in = mixed_7b_b2_b.shape[-1],
                           ch_out = 384
                          )
print(mixed_7b_b2_c_3x1)
mixed_7b_b2 = tf.concat(axis = 3,
                       values = [mixed_7b_b2_c_1x3, mixed_7b_b2_c_3x1])
print(mixed_7b_b2)

mixed_7b_b3_avg = tf.nn.avg_pool(mixed_7a,
                                 ksize = [1, 3, 3, 1],
                                 strides = [1, 1, 1, 1],
                                 padding =  'SAME',
                                 name = 'mixed_7b_b3_avg'
                                )
print(mixed_7b_b3_avg)
mixed_7b_b3_b = conv_layer(mixed_7b_b3_avg,
                           'mixed_7b_b3_b',
                           filt_shape = [1, 1],
                           ch_in = mixed_7b_b3_avg.shape[-1],
                           ch_out = 192
                          )
print(mixed_7b_b3_b)

mixed_7b = tf.concat(axis = 3,
                    values = [mixed_7b_b0, mixed_7b_b1, mixed_7b_b2, mixed_7b_b3_b])
print(mixed_7b)


#Now Mixed_7c

mixed_7c_b0 = conv_layer(mixed_7b,
                           'mixed_7c_b0',
                           filt_shape = [1, 1],
                           ch_in = mixed_7b.shape[-1],
                           ch_out = 320
                          )
print(mixed_7c_b0)

mixed_7c_b1_a = conv_layer(mixed_7b,
                           'mixed_7c_b1_a',
                           filt_shape = [1, 1],
                           ch_in = mixed_7b.shape[-1],
                           ch_out = 384
                          )
print(mixed_7c_b1_a)
mixed_7c_b1_b_1x3 = conv_layer(mixed_7c_b1_a,
                           'mixed_7c_b1_a_1x3',
                           filt_shape = [1, 3],
                           ch_in = mixed_7c_b1_a.shape[-1],
                           ch_out = 384
                          )
print(mixed_7c_b1_b_1x3)
mixed_7c_b1_b_3x1 = conv_layer(mixed_7c_b1_a,
                           'mixed_7c_b1_a_3x1',
                           filt_shape = [3, 1],
                           ch_in = mixed_7c_b1_a.shape[-1],
                           ch_out = 384
                          )
print(mixed_7c_b1_b_3x1)
mixed_7c_b1 = tf.concat(axis = 3,
                       values = [mixed_7c_b1_b_1x3, mixed_7c_b1_b_3x1])
print(mixed_7c_b1)

mixed_7c_b2_a = conv_layer(mixed_7b,
                           'mixed_7c_b2_a',
                           filt_shape = [1, 1],
                           ch_in = mixed_7b.shape[-1],
                           ch_out = 448
                          )
print(mixed_7c_b2_a)
mixed_7c_b2_b = conv_layer(mixed_7c_b2_a,
                           'mixed_7c_b2_b',
                           filt_shape = [3, 3],
                           ch_in = mixed_7c_b2_a.shape[-1],
                           ch_out = 384
                          )
print(mixed_7c_b2_b)

mixed_7c_b2_c_1x3 = conv_layer(mixed_7c_b2_b,
                           'mixed_7c_b2_c_1x3',
                           filt_shape = [1, 3],
                           ch_in = mixed_7c_b2_b.shape[-1],
                           ch_out = 384
                          )
print(mixed_7c_b2_c_1x3)
mixed_7c_b2_c_3x1 = conv_layer(mixed_7c_b2_b,
                           'mixed_7c_b2_c_3x1',
                           filt_shape = [3, 1],
                           ch_in = mixed_7c_b2_b.shape[-1],
                           ch_out = 384
                          )
print(mixed_7c_b2_c_3x1)
mixed_7c_b2 = tf.concat(axis = 3,
                       values = [mixed_7c_b2_c_1x3, mixed_7c_b2_c_3x1])
print(mixed_7c_b2)

mixed_7c_b3_avg = tf.nn.avg_pool(mixed_7b,
                                 ksize = [1, 3, 3, 1],
                                 strides = [1, 1, 1, 1],
                                 padding =  'SAME',
                                 name = 'mixed_7c_b3_avg'
                                )
print(mixed_7c_b3_avg)
mixed_7c_b3_b = conv_layer(mixed_7c_b3_avg,
                           'mixed_7c_b3_b',
                           filt_shape = [1, 1],
                           ch_in = mixed_7c_b3_avg.shape[-1],
                           ch_out = 192
                          )
print(mixed_7c_b3_b)

mixed_7c = tf.concat(axis = 3,
                    values = [mixed_7c_b0, mixed_7c_b1, mixed_7c_b2, mixed_7c_b3_b])
print(mixed_7c)


