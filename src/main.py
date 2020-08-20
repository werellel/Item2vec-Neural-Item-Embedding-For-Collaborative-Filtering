from item2vec import Item2Vec
from generate_data import generate_data
import tensorflow.compat.v1 as tf

from utils import save_data, read_data

tf.disable_v2_behavior()

DATA_PATH = 'generated_data.txt'

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

if __name__ == '__main__':
    user_nums = 1000
    item_nums = 300
    window_size = 10
    click_size = 20
    top_k = 10
    save_data(generate_data(user_nums = user_nums, 
                            item_nums = item_nums, 
                            window_size = window_size, 
                            click_size = click_size), 
              DATA_PATH)

    config = {}
    config['train_data_path'] = DATA_PATH # input data
    config['window'] = 10 # (maximum) window size
    config['embed_size'] = 100 # dimensionality of item embeddings
    config['alpha'] = 0.75 # smooth out unigram frequencies
    config['table_size'] = int(1E5) # table size from which to sample neg samples
    config['neg_sample_size'] = 15 # number of negative samples for each positive sample
    config['min_frequency'] = 0 #threshold for item frequency
    config['lr'] = 0.025 # initial learning rate
    config['min_lr'] = 0.001 # min learning rate
    config['epochs'] = 5 # number of epochs to train

    with tf.Session() as sess:
        i2v = Item2Vec(config, sess)
        i2v.preprocessing(config['train_data_path'])
        i2v.build_table()
        for idx in range(config['epochs']):
            print("epochs is ", idx)
            i2v.lr = config['lr']
            i2v.train_model(config['train_data_path'])
        print('Train Completed!')
        
        norm = tf.sqrt(tf.reduce_sum(tf.square(i2v.embed), 1, keepdims = True))
        normalized_embeddings = i2v.embed / norm
        normalized_embeddings = normalized_embeddings.eval(session=i2v.sess)  

        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

            plot_only = len(i2v.idx2item)
            low_dim_embs = tsne.fit_transform(normalized_embeddings[:plot_only,:])
            labels = [i2v.idx2item[i] for i in range(plot_only)]
            plot_with_labels(low_dim_embs, labels)

        except ImportError:
            print("Please install sklearn and matplotlib to visualize embeddings.")

        while True:
            print('Set item numbers. Range is {} to {}.'.format('0', int(item_nums)))
            try:
                input_data = int(input())
                i2v.get_sim_item([input_data], top_k)

            except Exception as e:
                print(e)
                continue
            