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
        batch_size = (tf.shape(features)[0])
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
                logits = self.mydecode_lstm(wordemb, nh, context,context2, reuse=(t!=0)).eval()
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