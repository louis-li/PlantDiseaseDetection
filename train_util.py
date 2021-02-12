import tensorflow as tf
import math
from tensorflow.keras.experimental import CosineDecay
from data_load import generator
import numpy as np
import datetime

def add_to_summary(summary_writer, loss, learning_rate, image1, image2, iteration):
    """Adds loss, learning_rate and images to tensorflow summary"""

    with summary_writer.as_default():
        tf.summary.scalar('Loss', loss, iteration)
        tf.summary.scalar('Learning_rate', learning_rate, iteration)
        tf.summary.image('image1', image1, iteration)
        tf.summary.image('image2', image2, iteration)
        
        
def nt_xent(z1, z2, batch_size, temperature, zdim):
    """Implements normalized temperature-scaled cross-entropy loss
    Args:
        z1: normalized latent representation of first set of augmented images [N, D]
        z2: normalized latent representation of second set of augmented images [N, D]
        batch_size: number of images in batch
        temperature: temperature for softmax. set in config
        zdim: dimension for latent representation set in config
    Returns:
        loss: contrastive loss averaged over batch (2*N samples)
    """

    # reshape so that the order is z1_1,z2_1,z1_2,z2_2,z1_3,z2_3
    z = tf.concat([z1,z2], axis=0)
    z_ = tf.reshape(tf.transpose(tf.reshape(z, [2, batch_size, zdim]), [1,0,2]), [batch_size*2, -1])

    # compute cosine similarity
    # a has order [z1_1*batch_size*2, z1_2*batch_size*2, ...]
    # b has order [z1_1, z1_2, z3_1 ...]
    a = tf.reshape(tf.transpose(tf.tile(tf.reshape(z_, [1, batch_size*2, zdim]), [batch_size*2 ,1, 1]), [1, 0, 2]), [batch_size*2*batch_size*2, zdim])
    b = tf.tile(z_, [batch_size*2, 1])
    sim = cosine_similarity(a, b)
    sim = tf.expand_dims(sim, axis=1)/temperature
    sim = tf.reshape(sim, [batch_size*2, batch_size*2])
    sim = tf.math.exp(sim-tf.reduce_max(sim))

    pos_indices = tf.concat([tf.range(1, (2*batch_size)**2, (batch_size*4)+2), tf.range(batch_size*2, (2*batch_size)**2, (batch_size*4)+2)], axis=0)
    pos_indices = tf.expand_dims(pos_indices, axis=1)
    pos_mask = tf.zeros(((2*batch_size)**2, 1), dtype=tf.int32)
    pos_mask = tf.tensor_scatter_nd_add(pos_mask, pos_indices, tf.ones((batch_size*2, 1), dtype=tf.int32))
    pos_mask = tf.reshape(pos_mask, [batch_size*2, batch_size*2])
    neg_mask = tf.ones((batch_size*2, batch_size*2), dtype=tf.int32) - tf.eye(batch_size*2, dtype=tf.int32)

    # similarity between z11-z12, z12-z11, z21-22, z22-z21 etc. 
    pos_sim = tf.reduce_sum(sim*tf.cast(pos_mask, tf.float32), axis=1) 

    # negative similarity consists of all similarities except i=j
    neg_sim = tf.reduce_sum(sim*tf.cast(neg_mask, tf.float32), axis=1)
    loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(pos_sim/neg_sim, 1e-10, 1.0)))

    return loss

def cosine_similarity(a, b):
    """Computes the cosine similarity between vectors a and b"""

    numerator = tf.reduce_sum(tf.multiply(a, b), axis=1)
    denominator = tf.multiply(tf.norm(a, axis=1), tf.norm(b, axis=1))
    cos_similarity = numerator/denominator
    return cos_similarity


def train(model, data, batch_size, warmup_epoch, total_epoch, lr, temperature, checkpoint_path = "models/checkpoints", model_summary_path="models/summary"):

    total_iterations = math.ceil(len(data) / batch_size)

    zdim = model.output.shape[1]
    checkpoint_path = f'models/{model.layers[1].name}_{datetime.date.today().strftime("%Y_%m_%d")}_checkpoints'

    lr_decayed_fn = tf.keras.experimental.CosineDecay( lr, total_iterations)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path , max_to_keep=10)

    summary_writer = tf.summary.create_file_writer(model_summary_path)

    images = generator(data, batch_size)
    
    # warm up head
    for e in range(warmup_epoch):
        epoch_loss = [] 
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        print(f"********************************************************************")
        print(f"**                        Warmup Epoch: {e}                         **")
        print(f"********************************************************************")
        for i in range(total_iterations):
            image1, image2 = next(images)
            # Train one step
            with tf.GradientTape() as tape:
                z1 = model(image1, training=True)
                z2 = model(image2, training=True)
                z1 = tf.math.l2_normalize(z1, axis=1)
                z2 = tf.math.l2_normalize(z2, axis=1)
                loss = nt_xent(z1, z2, batch_size,  temperature, zdim )
                reg_loss = tf.add_n(model.losses) if model.losses else 0
                loss = loss + reg_loss
            gradients = tape.gradient(loss, model.trainable_variables)

            # record loss
            epoch_loss.append(loss)

            # update optimizer
            optimizer.__setattr__('lr', lr_decayed_fn(i+1))

            # apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))        

            # 
            checkpoint.step.assign_add(1)
            if checkpoint.step.numpy() % 10 == 0:
                print(f"Iter: {i+2} Step: {checkpoint.step.numpy()} Loss: {loss.numpy():.5f} LR: {optimizer.__getattribute__('lr').numpy():9f}")
                add_to_summary(summary_writer, loss, optimizer.__getattribute__('lr'), image1[:1], image2[:1], checkpoint.step.numpy())
                summary_writer.flush()
        save_path = manager.save()
        print(f"Saved checkpoint for Epoch {e}: {save_path}")
        print("loss {:1.2f}".format(np.mean(epoch_loss)))

    # train all layers
    for l in model.layers:
        l.trainable = True
        
    for e in range(total_epoch):
        epoch_loss = [] 
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        print(f"********************************************************************")
        print(f"**                               Epoch: {e}                         **")
        print(f"********************************************************************")
        for i in range(total_iterations):
            image1, image2 = next(images)
            # Train one step
            with tf.GradientTape() as tape:
                z1 = model(image1, training=True)
                z2 = model(image2, training=True)
                z1 = tf.math.l2_normalize(z1, axis=1)
                z2 = tf.math.l2_normalize(z2, axis=1)
                loss = nt_xent(z1, z2, batch_size,  temperature, zdim )
                reg_loss = tf.add_n(model.losses) if model.losses else 0
                loss = loss + reg_loss
            gradients = tape.gradient(loss, model.trainable_variables)

            # record loss
            epoch_loss.append(loss)

            # update optimizer
            optimizer.__setattr__('lr', lr_decayed_fn(i+1))

            # apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))        

            # 
            checkpoint.step.assign_add(1)
            if checkpoint.step.numpy() % 10 == 0:
                print(f"Iter: {i+2} Step: {checkpoint.step.numpy()} Loss: {loss.numpy():.5f} LR: {optimizer.__getattribute__('lr').numpy():9f}")
                add_to_summary(summary_writer, loss, optimizer.__getattribute__('lr'), image1[:1], image2[:1], checkpoint.step.numpy())
                summary_writer.flush()
        save_path = manager.save()
        print(f"Saved checkpoint for Epoch {e}: {save_path}")
        #print("loss {:1.2f}".format(loss.numpy()))
        print("loss {:1.2f}".format(np.mean(epoch_loss)))
    return model