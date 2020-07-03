import tensorflow as tf
import os

def read_previous_performance_from_tfevent(model_id,epoch,split='dev',metric='Accuracy'):
    # get name of file in TF log dir
    tf_log_dir = 'data/models/model_{}/{}/'.format(model_id,split)
    eventfile = os.listdir(tf_log_dir)
    assert len(eventfile)==1
    for summary in tf.train.summary_iterator(tf_log_dir+eventfile[0]):
        # find epoch
        if summary.step==epoch:
            for value in summary.summary.value:
                # find metric
                if value.tag == 'evaluation_metrics/{}'.format(metric):
                    return value.simple_value

if __name__ == '__main__':

    for epoch in range(5,45,5):
        print('epoch {}: {}'.format(epoch,read_previous_performance_from_tfevent(3545, epoch, split='dev',metric='Accuracy')))
