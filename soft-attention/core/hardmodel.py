# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf


class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[49, 128], dim_embed=512, dim_hidden=1024, n_time_step=16,
                  prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(tf.nn.dropout(features_proj,0.5) + tf.expand_dims(tf.matmul(tf.nn.dropout(h,0.5), w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha

 
    def myattention_layer(self, features, h, reuse=False):
        with tf.variable_scope('myattention_layer', reuse=reuse):
            features_mean = tf.reshape(tf.reduce_mean(features,1, name='featuremean'),[-1,self.D,1])
            w1_1 = tf.get_variable('w1_1', [1, self.M], initializer=self.weight_initializer)
            b1_1 = tf.get_variable('b1_1', [self.M], initializer=self.const_initializer)
            pctx_1 = tf.tensordot(features_mean, w1_1,[[2],[0]])+b1_1
            w1_2 = tf.get_variable('w1_2', [self.H, self.M], initializer=self.weight_initializer)
            pstate_1 = tf.expand_dims(tf.matmul(h, w1_2),1)
            pctx1 = tf.nn.tanh(pctx_1+pstate_1)
            w1_3 = tf.get_variable('w1_3', [self.M, 1], initializer=self.weight_initializer)
            b1_2 = tf.get_variable('b1_2', [1], initializer=self.const_initializer)
            alpha1 = tf.reshape(tf.tensordot(pctx1,w1_3,[[2],[0]])+b1_2, [-1, self.D])
            alpha1 = tf.nn.softmax(alpha1)  ## N * 512
            weighted_features = tf.multiply(features, tf.expand_dims(alpha1,1))*512
            w2_1 = tf.get_variable('w2_1', [self.D, self.M], initializer=self.weight_initializer)
            b2_1 = tf.get_variable('b2_1', [self.M], initializer=self.const_initializer)
            pctx_2 = tf.tensordot(weighted_features, w2_1,[[2],[0]])+b2_1
            w2_2 = tf.get_variable('w2_2', [self.H, self.M], initializer=self.weight_initializer)
            pstate_2 = tf.expand_dims(tf.matmul(h, w2_2),1)
            pctx2 = tf.nn.tanh(pctx_2+pstate_2)
            w2_3 = tf.get_variable('w2_3', [self.M, 1], initializer=self.weight_initializer)
            b2_2 = tf.get_variable('b2_2', [1], initializer=self.const_initializer)
            alpha2 = tf.reshape(tf.tensordot(pctx2,w2_3,[[2],[0]])+b2_2, [-1, self.L])
            alpha2 = tf.nn.softmax(alpha2)  ### N* 196
            alpha1_reshape = tf.expand_dims(alpha1,1)
            alpha2_reshape = tf.expand_dims(alpha2,2)
            context = tf.multiply(tf.multiply(features,alpha1_reshape),alpha2_reshape)*512*196
            context2 = tf.reshape(tf.reduce_mean(context,1),[-1,self.D], name='context')
            contextnorm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(context2),1),[-1,1]))
            contextnorm = contextnorm+ tf.constant([1e-8])
            context_normal = tf.divide(context2,contextnorm)
            context3 = tf.reshape(tf.reduce_mean(context,2),[-1,self.L], name='context3')
            contextnorm2 = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(context3),1),[-1,1]))
            contextnorm2 = contextnorm2+ tf.constant([1e-8])
            context_normal2 = tf.divide(context3,contextnorm2)       
            return context_normal,context_normal2, alpha2, alpha1

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def mydecode_lstm(self, x, h, context,context2, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)
            wx1 = tf.get_variable('wx1', [self.M, self.M], initializer=self.weight_initializer)
            wh1 = tf.get_variable('wh1', [self.H, self.M], initializer=self.weight_initializer)
            wx2 = tf.get_variable('wx2', [self.M, self.M], initializer=self.weight_initializer)
            wh2 = tf.get_variable('wh2', [self.H, self.M], initializer=self.weight_initializer)
            b1 = tf.get_variable('b1', [self.M], initializer=self.const_initializer)
            b2 = tf.get_variable('b2', [self.M], initializer=self.const_initializer)
            wx3 = tf.get_variable('wx3', [self.M, self.M], initializer=self.weight_initializer)
            wh3 = tf.get_variable('wh3', [self.H, self.M], initializer=self.weight_initializer)
            b3 = tf.get_variable('b3', [self.M], initializer=self.const_initializer)
            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.nn.relu(tf.matmul(h, w_h) + b_h)

            vgate = tf.nn.sigmoid(tf.matmul(tf.nn.dropout(x,0.5), wx1)+ tf.matmul(h,wh1)+b1)
            #vgate = tf.nn.sigmoid(tf.matmul(x, wx1)+ tf.matmul(h,wh1)+b1)
            v2gate = tf.nn.sigmoid(tf.matmul(tf.nn.dropout(x,0.5), wx3)+ tf.matmul(h,wh3)+b3)
            #v2gate = tf.nn.sigmoid(tf.matmul(x, wx3)+ tf.matmul(h,wh3)+b3)
            w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
            w2_ctx2out = tf.get_variable('w2_ctx2out', [self.L, self.M], initializer=self.weight_initializer)
            contextvec = tf.multiply(tf.matmul(tf.nn.dropout(context,0.5), w_ctx2out),vgate)
            #contextvec = tf.multiply(tf.matmul(context, w_ctx2out),vgate)
            #contextvec2 = tf.multiply(tf.matmul(context2, w2_ctx2out),v2gate)
            contextvec2 = tf.multiply(tf.matmul(tf.nn.dropout(context2,0.5), w2_ctx2out),v2gate)
            xgate = tf.nn.sigmoid(tf.matmul(tf.nn.dropout(x,0.5), wx2)+ tf.matmul(h,wh2)+b2)
            #xgate = tf.nn.sigmoid(tf.matmul(x, wx2)+ tf.matmul(h,wh2)+b2)
            #xvec = tf.multiply(tf.nn.dropout(x,0.5),xgate)
            xvec = tf.multiply(x,xgate)
            
            #if self.prev2out:
            #    h_logits += x
            #h_logits = tf.concat([h_logits, contextvec, contextvec2, xvec],axis=1)
            h_logits += contextvec + contextvec2 +xvec*0
            #h_logits = tf.nn.tanh(h_logits)
            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))


        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)
        #features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        alpha2_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            #precontext, prealpha = self._attention_layer(features,features_proj, h, reuse=(t!=0))

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=x[:,t,:], state=[c, h])
                #_, (c, h) = lstm_cell(inputs=tf.concat([x[:,t,:],precontext],1), state=[c, h])

            context, context2, alpha, alpha2 = self.myattention_layer(features, h, reuse=(t!=0))
            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
            alpha_list.append(alpha)
            alpha2_list.append(alpha2)
            logits = self.mydecode_lstm(x[:,t,:], h, context,context2, dropout=self.dropout, reuse=(t!=0))
            #logits = self._decode_lstm(x[:,t,:], h, context, dropout=self.dropout, reuse=(t!=0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=captions_out[:, t]) * mask[:, t])

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
            alphas_all = tf.reduce_mean(alphas, 1)      # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((16./196- alphas_all) ** 2)
            alphas2 = tf.transpose(tf.stack(alpha2_list), (1, 0, 2))     # (N, T, D)
            alphas2_all = tf.reduce_mean(alphas2, 1)      # (N, D)
            alpha_reg2 = self.alpha_c * tf.reduce_sum((16./512- alphas2_all) ** 2)
            loss += alpha_reg+alpha_reg2

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        #features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
	alpha2_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)


            #precontext, prealpha = self._attention_layer(features,features_proj, h, reuse=(t!=0))

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=x, state=[c, h])
                #_, (c, h) = lstm_cell(inputs=tf.concat([x,precontext],1), state=[c, h])
            #with tf.variable_scope('lstm', reuse=(t!=0)):
            #    _, (c, h) = lstm_cell(inputs=x, state=[c, h])

            context, context2, alpha, alpha2 = self.myattention_layer(features, h, reuse=(t!=0))
            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
                beta_list.append(beta)
            alpha_list.append(alpha)
	    alpha2_list.append(alpha2)
            logits = self.mydecode_lstm(x, h, context,context2, reuse=(t!=0))
            #logits = self._decode_lstm(x, h, context, reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        #betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        betas = beta_list
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions

    def mybuild_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        #features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        alpha2_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        batch_size = (tf.shape(features)[0])#.eval()
        beam_size = 3
        word_pred_list = []  # beam_size * batch_size   [[word], 1]
        state_list = []
        for i in range(beam_size):
            word_pred_list.append([[[self._start],1]]*batch_size)
            state_list.append([c,h])


        for t in range(maxlen):

            context, context2, alpha, alpha2 = self.myattention_layer(features, h, reuse=(t!=0))
            alpha_list.append(alpha)
            alpha2_list.append(alpha2)
            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
            beta_list.append(beta)
            newstate_list = []
            logits_list = []      
            for i in range(beam_size):
                x = [each[0][-1] for each in word_pred_list[i]]
                wordemb = self._word_embedding(inputs=tf.reshape(tf.convert_to_tensor(x,dtype=tf.int32),[batch_size,1]), reuse=(t!=0))
                with tf.variable_scope('lstm', reuse=(t!=0)):
                    _, (nc, nh) = lstm_cell(inputs=tf.concat([wordemb, context, context2], 1), state=state_list[i])
                newstate_list.append([nc,nh])
                logits = self.mydecode_lstm(wordemb, nh, context,context2, reuse=(t!=0))#.eval()
                logits_list.append(logits)
            logits_list = np.array(logits_list).transpose([1,0,2]).reshape([batch_size,-1])
            logits_pick = np.argsort(logits_list)[:,-beam_size]
            new_word_pred_list =[[],[],[]]
            for i in range(beam_size):
                for j in range(batch_size):
                    idx = logits_pick[j][i]
                    whichbeam = int(idx/self.V)
                    state_list[i][0][j]=newstate_list[whichbeam][0][j]
                    state_list[i][1][j]=newstate_list[whichbeam][1][j]
                    prev = word_pred_list[whichbeam][j]
                    cur_word = idx%self.V
                    cur_prob = logits_list[j][idx]
                    cur = [prev[0].append(cur_word), prev[1]+cur_prob]
                    new_word_pred_list[i].append(cur)
            word_pred_list = new_word_pred_list

        sampled_captions = []
        for i in range(batch_size):
            tmplist = []
            for j in range(beam_size):
                tmplist.append(word_pred_list[j][i])
            tmplist = sorted(tmplist, reverse=False, key=lambda l: l[1])
            sampled_captions.append(tmplist[-1][0])

        sampled_captions = tf.reshape(tf.convert_to_tensor(sampled_captions,dtype=tf.int32),[batch_size,max_len])
        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        #betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        betas = beta_list
        #sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions


    def predict(self, isreuse, _state, features_bn,  beam_size, word, lstm_cell):
        batch_size = len(word)
        state = []
        for s in _state:
            state.append(tf.reshape(tf.convert_to_tensor(s,dtype=tf.float32),[batch_size,self.H]))
        context, context2, alpha, _ = self.myattention_layer(features_bn, state[1], reuse=isreuse)
        if self.selector:
            context, beta = self._selector(context, state[1], reuse=isreuse)
        
        wordemb = self._word_embedding(inputs=tf.reshape(tf.convert_to_tensor(word,dtype=tf.int32),[batch_size]), reuse=isreuse)
        with tf.variable_scope('lstm', reuse=isreuse):
                _, (nc, nh) = lstm_cell(inputs=tf.concat([wordemb, context, context2], 1), state=state)
        logits = self.mydecode_lstm(wordemb, nh, context,context2, reuse=isreuse)
        logits = tf.nn.softmax(logits)
	newstate = [nc,nh]
        return logits, newstate, alpha, beta
