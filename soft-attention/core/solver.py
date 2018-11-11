import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def train(self):
        # train/val dataset
        n_examples = self.data['features'].shape[0]
        #n_examples = 5000
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        val_features = self.val_data['features']
        n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

        # build graphs for training model and sampling captions
        # This scope fixed things!!
        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.model.build_sampler(max_len=20)
            #_, _, mygenerated_captions = self.model.mybuild_sampler(max_len=20)

        # train op
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op
        # tf.scalar_summary('batch_loss', loss)
        #tf.summary.scalar('batch_loss', loss)
        #for var in tf.trainable_variables():
            #tf.histogram_summary(var.op.name, var)
        #    tf.summary.histogram(var.op.name, var)
        #for grad, var in grads_and_vars:
            #tf.histogram_summary(var.op.name+'/gradient', grad)
        #    tf.summary.histogram(var.op.name+'/gradient', grad)

        #summary_op = tf.merge_all_summaries()
        #summary_op = tf.summary.merge_all()

        print "Epoch is: %d" %self.n_epochs
        print "Number of images: %d" %n_examples
        print "Batch size is: %d" %self.batch_size
        print "Iterations in each epoch: %d" %n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            #summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            #summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Restore pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[image_idxs_batch]
                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    # write summary for tensorboard visualization
                    #if i % 10 == 0:
                    #    summary = sess.run(summary_op, feed_dict)
                    #    summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

                    if (i+1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" %(j+1, gt)
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" %decoded[0]

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "It took time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0
                use_beam = False
                # print out BLEU scores and file write
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 20))
                    for i in range(n_iters_val):
                        features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                        #feed_dict = {self.model.features: features_batch}
                        #gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        if use_beam:
                           _,_ ,gen_cap = self.beamsearch(features_batch, 20, sess)
                        else:
                           feed_dict = {self.model.features: features_batch}
                           gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
	                #print "cap shape", gen_cap.shape	
			all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    scores = evaluate(data_path='./data', split='val', get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)

                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." %(e+1)
                


    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = { self.model.features: features_batch }
            use_beam = False
            if use_beam:
                alps,bts ,sam_cap = self.beamsearch(features_batch, 20, sess)
            else:
                feed_dict = {self.model.features: features_batch}
                alps, bts, sam_cap = sess.run([alpha, betas, sampled_captions], feed_dict=feed_dict)
            #alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.model.features: features_batch }
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" %(split,split))

    def beamsearch(self, features_batch, maxlen, sess):
        tf.get_variable_scope().reuse_variables()
	features = self.model._batch_norm(self.model.features, mode='test', name='conv_features')
        feed_dict = {self.model.features: features_batch}
        _c, _h = self.model._get_initial_lstm(features=features)
        c,h = sess.run([_c, _h],feed_dict=feed_dict)
        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.model.H)
        batch_size = len(features_batch)
        beam_size = 3
        word_pred_list = []  # beam_size * batch_size   [[word], 1]
        state_list = []


        for i in range(beam_size):
            word_pred_list.append([[[self.model._start],1]]*batch_size)

            state_list.append([c,h])
        for t in range(maxlen):
            newstate_list=[]
            logit_list=[]
            tmpalpha = None
            tmpbeta = None
            for i in range(beam_size):
		#print "wordlistlen", len(word_pred_list)
		#print len(word_pred_list[i])
		#print word_pred_list[i][0]
                word = [each[0][-1] for each in word_pred_list[i]]
                _logit, _newstate, _alpha, _beta = self.model.predict(t!=0, state_list[i], features, beam_size, word, lstm_cell)
                logit, newstate, alpha, beta = sess.run([_logit, _newstate, _alpha, _beta],feed_dict=feed_dict)
                logit_list.append(logit)
                newstate_list.append(newstate)
                tmpalpha=alpha
                tmpbeta=beta
            alpha_list.append(alpha)
            beta_list.append(beta)

            if t==0:
                logit_list = np.array(logit_list[0]).reshape([batch_size,-1])
            else:
                logit_list = np.array(logit_list).transpose([1,0,2]).reshape([batch_size,-1])
            logits_pick = np.argsort(logit_list)[:,-beam_size:]
            new_word_pred_list =[[],[],[]]
	    #print "batch_szie",batch_size
            #print logits_pick.shape
            #print "before update"
            #print word_pred_list[0][0]
            import copy
            for i in range(beam_size):
                for j in range(batch_size):
                    idx = logits_pick[j][i]
	            #print "idx",idx
                    whichbeam = int(idx/self.model.V)
		    #print "whichbeam",whichbeam
                    state_list[i][0][j]=newstate_list[whichbeam][0][j]
                    state_list[i][1][j]=newstate_list[whichbeam][1][j]
                    prev = word_pred_list[whichbeam][j]
                    #print "prev",prev
                    cur_word = idx%self.model.V
                    #print "cur_word", cur_word
                    cur_prob = logit_list[j][idx]
		    tmp = copy.deepcopy(prev[0])
		    tmp.append(cur_word)
                    cur = [tmp, prev[1]+cur_prob]
                    #print "cur",cur
                    new_word_pred_list[i].append(cur)
            word_pred_list = new_word_pred_list
            #print "after update"
            #print word_pred_list[0][0]
        #print "check word_pred_list"
        #print len(word_pred_list)
        #print len(word_pred_list[0])
        #print len(word_pred_list[0][0])
        #for i in range(beam_size):
	#   for j in range(10):
	#      print word_pred_list[i][j]
        sampled_captions = []
        for i in range(batch_size):
            tmplist = []
            for j in range(beam_size):
                tmplist.append(word_pred_list[j][i])
            tmplist = sorted(tmplist, reverse=False, key=lambda l: l[1])
            #print "put in the cap", tmplist[-1][0]
            sampled_captions.append(tmplist[-1][0][1:])

        sampled_captions = np.array(sampled_captions)
        alphas = np.array(alpha_list).transpose(1,0,2)    # (N, T, L)
        #betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        betas = beta_list
        #sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions 	
