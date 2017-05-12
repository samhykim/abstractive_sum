import tensorflow as tf

k = 10
class BeamSearch:


  def __init__(self, model):
    self.model = model

  def beam_search_single_step(self, model, time_step, top_hypotheses, values, indices,
                              cell, state, dropout_rate, old_states, scope):
    if time_step == 0:
      word_inds = [model.vocab.START for _ in range(model.config.batch_size)]         
      word_embeddings = tf.reshape(tf.nn.embedding_lookup(model.label_embeddings, word_inds),
                                  (model.config.batch_size, model.config.embed_size))


      outputs, state_t = model.get_output(word_embeddings, cell, state, dropout_rate, old_states, scope)
      outputs = tf.log(tf.nn.softmax(outputs)) # normalize

      values, indices = tf.nn.top_k(outputs, k) # start with top k outputs

      # reshape top hypothesis to (batch_size, k, 1) for concatenation later
      top_hypotheses = tf.reshape(indices, [-1, k, 1])
    else:
      new_cs = []
      new_ms = []
      for j in range(k):
        word_embeddings = tf.reshape(tf.nn.embedding_lookup(model.label_embeddings, indices[:,j]), 
                                    (model.config.batch_size, model.config.embed_size))

        if time_step == 1:
          outputs, state_t = model.get_output(word_embeddings, cell, state, dropout_rate, old_states, scope)
        else:
          input_state = (state[0][:,j,:], state[1][:,j,:])
          outputs, state_t = model.get_output(word_embeddings, cell, input_state, dropout_rate, old_states, scope)
        new_cs.append(state_t[0]) # (batch_size, lstm_num_units)
        new_ms.append(state_t[1])
        #print state_t[0].get_shape().as_list()
        #print state_t[0].get_shape().as_list()
        outputs = tf.log(tf.nn.softmax(outputs)) # normalize

        next_values, next_indices = tf.nn.top_k(outputs, k)
        next_values = tf.tile(tf.reshape(values[:,j], (-1, 1)), [1, k]) + next_values
        if j == 0:
          all_values, all_indices = next_values, next_indices
        else:
          all_values, all_indices = tf.concat(1, [all_values, next_values]), tf.concat(1, [all_indices, next_indices])

      values, next_indices = tf.nn.top_k(all_values, k) # top k hypothesis
      new_cs = tf.stack(new_cs, 1) # (batch_size, k, lstm_num_units)
      new_ms = tf.stack(new_ms, 1)
      # grab the indices of the top k tokens
      indices = []
      for i in range(model.config.batch_size):
        indices.append(tf.gather(all_indices[i,:], next_indices[i,:]))
      indices = tf.stack(indices, 0)

      # grab the index of previous time_step
      prev_indices = next_indices / k 

      #top_hypotheses = tf.concat(1, [top_hypotheses, tf.reshape(indices, [-1, 1, k])])
      new_hypotheses = []
      new_state_cs, new_state_ms = [], []
      for i in range(model.config.batch_size):
        row = tf.gather_nd(top_hypotheses[i,:,:], tf.reshape(prev_indices[i,:], [-1,1]))
        state_cs = tf.gather_nd(new_cs[i,:,:], tf.reshape(prev_indices[i,:], [-1,1]))
        state_ms = tf.gather_nd(new_ms[i,:,:], tf.reshape(prev_indices[i,:], [-1,1]))
        new_hypotheses.append(row)
        new_state_cs.append(state_cs)
        new_state_ms.append(state_ms)
      top_hypotheses = tf.stack(new_hypotheses, 0)
      new_cs = tf.stack(new_state_cs, 0)
      new_ms = tf.stack(new_state_ms, 0)

      state_t = (new_cs, new_ms)

      top_hypotheses = tf.concat(2, [top_hypotheses, tf.reshape(indices, [-1, k, 1])])
      print top_hypotheses.get_shape().as_list()

      assert indices.get_shape().as_list() == [model.config.batch_size, k]

    return top_hypotheses, values, indices, state_t

  def beam_search(self, model, time_step, top_hypotheses, values, indices,
                              cell, state, dropout_rate, old_states, scope):

    word_inds = [model.vocab.START for _ in range(model.config.batch_size)]         
    word_embeddings = tf.reshape(tf.nn.embedding_lookup(model.label_embeddings, word_inds),
                                (model.config.batch_size, model.config.embed_size))


    outputs, state = model.get_output(word_embeddings, cell, state, dropout_rate, old_states, scope)
    outputs = tf.log(tf.nn.softmax(outputs)) # normalize

    values, indices = tf.nn.top_k(outputs, k) # start with top k outputs

    # reshape top hypothesis to (batch_size, k, 1) for concatenation later
    top_hypotheses = tf.reshape(indices, [-1, k, 1])


    for time_step in range(1, model.config.max_dec_length):
      new_cs = []
      new_ms = []
      for j in range(k):
        word_embeddings = tf.reshape(tf.nn.embedding_lookup(model.label_embeddings, indices[:,j]), 
                                    (model.config.batch_size, model.config.embed_size))

        if time_step == 1:
          outputs, state = model.get_output(word_embeddings, cell, state, dropout_rate, old_states, scope)
        else:
          input_state = (state[0][:,j,:], state[1][:,j,:])
          outputs, state_t = model.get_output(word_embeddings, cell, input_state, dropout_rate, old_states, scope)
        new_cs.append(state_t[0]) # (batch_size, lstm_num_units)
        new_ms.append(state_t[1])
        #print state_t[0].get_shape().as_list()
        #print state_t[0].get_shape().as_list()
        outputs = tf.log(tf.nn.softmax(outputs)) # normalize

        next_values, next_indices = tf.nn.top_k(outputs, k)
        next_values = tf.tile(tf.reshape(values[:,j], (-1, 1)), [1, k]) + next_values
        if j == 0:
          all_values, all_indices = next_values, next_indices
        else:
          all_values, all_indices = tf.concat(1, [all_values, next_values]), tf.concat(1, [all_indices, next_indices])

      values, next_indices = tf.nn.top_k(all_values, k) # top k hypothesis
      new_cs = tf.stack(new_cs, 1) # (batch_size, k, lstm_num_units)
      new_ms = tf.stack(new_ms, 1)
      # grab the indices of the top k tokens
      indices = []
      for i in range(model.config.batch_size):
        indices.append(tf.gather(all_indices[i,:], next_indices[i,:]))
      indices = tf.stack(indices, 0)

      # grab the index of previous time_step
      prev_indices = next_indices / k 

      #top_hypotheses = tf.concat(1, [top_hypotheses, tf.reshape(indices, [-1, 1, k])])
      new_hypotheses = []
      new_state_cs, new_state_ms = [], []
      for i in range(model.config.batch_size):
        row = tf.gather_nd(top_hypotheses[i,:,:], tf.reshape(prev_indices[i,:], [-1,1]))
        state_cs = tf.gather_nd(new_cs[i,:,:], tf.reshape(prev_indices[i,:], [-1,1]))
        state_ms = tf.gather_nd(new_ms[i,:,:], tf.reshape(prev_indices[i,:], [-1,1]))
        new_hypotheses.append(row)
        new_state_cs.append(state_cs)
        new_state_ms.append(state_ms)
      top_hypotheses = tf.stack(new_hypotheses, 0)
      new_cs = tf.stack(new_state_cs, 0)
      new_ms = tf.stack(new_state_ms, 0)

      state = (new_cs, new_ms)

      top_hypotheses = tf.concat(2, [top_hypotheses, tf.reshape(indices, [-1, k, 1])])
      print top_hypotheses.get_shape().as_list()

      assert indices.get_shape().as_list() == [model.config.batch_size, k]

    preds = []
    top_values, top_indices = tf.nn.top_k(values) # (batch_size, 1)
    for i in range(model.config.batch_size):
        preds.append(top_hypotheses[i, top_indices[i,0], :])
    
    #top_hypothesis, _ = tf.nn.top_k(tf.reshape(top_hypotheses, [self.config.batch_size, self.config.max_dec_length,-1]), 1) # get best hypothesis
    #preds = tf.reshape(top_hypothesis, [-1, self.config.max_dec_length])
    preds = tf.stack(preds, axis=0)

    return preds




  def beam_search_old(self, *args):
    
    word_inds = [self.model.labels_vocab.START for _ in range(self.model.config.batch_size)]         
    word_embeddings = tf.reshape(tf.nn.embedding_lookup(self.model.label_embeddings, word_inds),
                                (self.model.config.batch_size, self.model.config.embed_size))

    outputs, _ = self.model.get_output(word_embeddings, *args)

    values, indices = tf.nn.top_k(outputs, k) # start with top k outputs
    values = tf.log(tf.nn.softmax(values)) # normalize

    assert indices.get_shape().as_list() == [batch_size, k]
    #top_hypotheses = []
    # for i in range(batch_size):
    #   for j in range(k):
    #     row = []
    #     row.append(tf.reshape(indices[i,j], [1])) # reshape scalar to vector
    #   top_hypotheses.append(row)
    top_hypotheses = tf.reshape(indices, [-1, k, 1])


    for time_step in range(1, self.model.config.max_dec_length):
      for j in range(k):
        word_embeddings = tf.reshape(tf.nn.embedding_lookup(self.model.label_embeddings, indices[:,j]), 
                                    (self.model.config.batch_size, self.model.config.embed_size))

        outputs, _ = self.model.get_output(word_embeddings, *args)

        next_values, next_indices = tf.nn.top_k(outputs, k)
        next_values = tf.log(tf.nn.softmax(next_values)) # normalize
        next_values = tf.tile(tf.reshape(values[:,j], (-1, 1)), [1, k]) + next_values
        if j == 0:
          all_values, all_indices = next_values, next_indices
        else:

          all_values, all_indices = tf.concat(1, [all_values, next_values]), tf.concat(1, [all_indices, next_indices])

        #all_values.append(tf.tile(tf.reshape(values[:,j], (-1, 1)), [k]) + next_values)
        #all_indices.append(next_indices)

      # all_values, all_indices should have shape = (None, k*k)

      #all_values = tf.stack(all_values, axis=1) # shape = (None, k*k)
      #all_indices = tf.stack(all_indices, axis=1) # shape = (None, k*k)

      values, next_indices = tf.nn.top_k(all_values, k) # top k hypothesis
      
      #print all_indices.get_shape().as_list()
      #print next_indices.get_shape().as_list()
      #all_indices = tf.reshape(all_indices, [batch_size, -1])
      #next_indices = tf.reshape(next_indices, [batch_size, -1])
      #all_indices, next_indices = tf.transpose(all_indices), tf.transpose(next_indices)
      indices = []
      for i in range(self.model.config.batch_size):
        indices.append(tf.gather(all_indices[i,:], next_indices[i,:]))

      indices = tf.stack(indices, 0)
      #print indices

      prev_indices = next_indices / k # grab the index of previous time_step

      new_hypotheses = []
      for i in range(self.model.config.batch_size):
        row = tf.gather_nd(top_hypotheses[i,:,:], tf.reshape(prev_indices[i,:], [-1,1]))
        new_hypotheses.append(row)
      top_hypotheses = tf.stack(new_hypotheses, 0)
             

      top_hypotheses = tf.concat(2, [top_hypotheses, tf.reshape(indices, [-1, k, 1])])

      # new_hypotheses = []
      # for i in range(batch_size):
      #   for j in range(k):
      #     row = []
      #     # concatenate previous sequence to current index
      #     row.append(tf.concat(1, [tf.gather_nd(top_hypotheses[i], [prev_indices[i,j]]), indices[i,j]])) 
      #   new_hypotheses.append(row)
      #top_hypotheses = tf.concat(1, [top_hypotheses, tf.reshape(indices, [-1, 1, k])])
      print top_hypotheses.get_shape().as_list()
      #top_hypotheses = new_hypotheses


    top_hypothesis, _ = tf.nn.top_k(tf.reshape(top_hypotheses, [-1, self.model.config.max_dec_length,k]), 1) # get best hypothesis
    print top_hypothesis.get_shape.as_list()
    top_hypothesis = tf.reshape(top_hypothesis, [-1, self.model.config.max_dec_length])
    print top_hypothesis.get_shape().as_list()
    return top_hypothesis

