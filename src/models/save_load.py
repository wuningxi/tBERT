import os
import tensorflow as tf
import shutil
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

def get_model_dir(opt,VM_copy=False):
    try:
        if type(opt['id'])==str:
            model_folder = opt['datapath'] + 'baseline_models/SemEval2017_task3_submissions_and_scores 3/' + opt['id'] + '/'
        else:
            if VM_copy:
                model_folder = opt['datapath'] + 'VM_models/model_{}/'.format(opt['id'])
            else:
                model_folder = opt['datapath'] + 'models/model_{}/'.format(opt['id'])
    except KeyError:
        raise KeyError('"id" and "datapath" in opt dictionary necessary for saving or loading model.')
    return model_folder

def load_model(opt, saver, sess, epoch):
    # for early stopping
    model_path = get_model_dir(opt) + 'model_epoch{}.ckpt'.format(epoch)
    saver.restore(sess,model_path)

# def load_predictions(model_id):
#     opt = {'datapath': 'data/','id':model_id}
#     model_dir = get_model_dir(opt,VM_copy=True)
#     pred_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) and f.endswith('.pred')]
#     read_predictions(opt, subset='dev', VM_path=True)
#     pass

def load_model_and_graph(opt, sess, epoch, vm_path=True):
    # for model inspection
    model_path = get_model_dir(opt,VM_copy=vm_path) + 'model_epoch{}.ckpt'.format(epoch)
    new_saver = tf.train.import_meta_graph(model_path+'.meta') # load graph
    new_saver.restore(sess,model_path) # restore weights

def run_restored_tensor(tensornames, left, right, sess=None, label=None, w_tl=None, w_tr=None, d_tl=None, d_tr=None, t_l=None, t_r=None):
    '''
    Runs graph on specified tensor(s) with supplied word ids for left and right sentence for inspection
    :param tensornames: names of tensors to restore
    :param left: numpy array of word ids for left sentence
    :param right: numpy array of word ids for right sentence
    :param sess: session object
    :return: values of evaluated tensors
    '''
    graph = tf.get_default_graph()
    # build feed_dict based on provided input
    XL = graph.get_tensor_by_name("XL:0")
    XR = graph.get_tensor_by_name("XR:0")
    feed_dict = {XL: left, XR: right}
    if not label is None:
        Y = graph.get_tensor_by_name("labels:0")
        feed_dict[Y]=label
    if not d_tl is None:
        D_TL = graph.get_tensor_by_name("D_TL:0")
        D_TR = graph.get_tensor_by_name("D_TR:0")
        feed_dict[D_TL]=d_tl
        feed_dict[D_TR]=d_tr
    if not w_tl is None:
        W_TL = graph.get_tensor_by_name("W_TL:0")
        W_TR = graph.get_tensor_by_name("W_TR:0")
        feed_dict[W_TL]=w_tl
        feed_dict[W_TR]=w_tr
    if not t_l is None:
        T_L = graph.get_tensor_by_name("TL:0")
        T_R = graph.get_tensor_by_name("TR:0")
        feed_dict[T_L]=t_l
        feed_dict[T_R]=t_r
    # restore tensors
    ops_to_restore = []
    for tensor in tensornames:
        ops_to_restore.append(graph.get_tensor_by_name(tensor+":0"))
    return sess.run(ops_to_restore, feed_dict)

def run_restored_bert_tensor(tensornames, word_ids, mask_ids, seg_ids, sess=None, label=None, w_tl=None, w_tr=None, d_tl=None, d_tr=None):
    '''
    Runs graph on specified tensor(s) with supplied word ids for left and right sentence for inspection
    :param tensornames: names of tensors to restore
    :param left: numpy array of word ids for left sentence
    :param right: numpy array of word ids for right sentence
    :param sess: session object
    :return: values of evaluated tensors
    '''
    graph = tf.get_default_graph()
    # build feed_dict based on provided input
    X = graph.get_tensor_by_name("Placeholder:0")
    X_mask = graph.get_tensor_by_name("Placeholder_1:0")
    X_seg = graph.get_tensor_by_name("Placeholder_2:0")

    feed_dict = {X: word_ids, X_mask: mask_ids, X_seg: seg_ids}
    if not label is None:
        Y = graph.get_tensor_by_name("labels:0")
        feed_dict[Y]=label
    if not d_tl is None:
        D_TL = graph.get_tensor_by_name("D_TL:0")
        D_TR = graph.get_tensor_by_name("D_TR:0")
        feed_dict[D_TL]=d_tl
        feed_dict[D_TR]=d_tr
    if not w_tl is None:
        W_TL = graph.get_tensor_by_name("W_TL:0")
        W_TR = graph.get_tensor_by_name("W_TR:0")
        feed_dict[W_TL]=w_tl
        feed_dict[W_TR]=w_tr
    # restore tensors
    ops_to_restore = []
    for tensor in tensornames:
        ops_to_restore.append(graph.get_tensor_by_name(tensor+":0"))
    return sess.run(ops_to_restore, feed_dict)

def create_saver():
    return tf.train.Saver(max_to_keep=1)

def create_model_folder(opt):
    folder = get_model_dir(opt)
    if os.path.exists(folder):
        FileExistsError('{} already exists. Please delete.'.format(folder))
    else:
        os.mkdir(folder)

def save_model(opt, saver, sess, epoch):
    model_path = get_model_dir(opt) + 'model_epoch{}.ckpt'.format(epoch)
    print(model_path)
    saver.save(sess, model_path)

def delete_all_checkpoints_but_best(opt,best_epoch):
    # list all files in model dir
    model_dir = get_model_dir(opt)
    # list all checkpoints but best
    best_model = 'model_epoch{}.ckpt'.format(best_epoch)
    to_delete = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) and f.startswith('model_epoch') and not f.startswith(best_model)]
    if len(to_delete)>0:
        print('Deleting the following checkpoint files:')
        for f in to_delete:
            file_path = os.path.join(model_dir, f)
            print(file_path)
            # delete
            os.remove(file_path)

def delete_model_dir(opt):
    model_dir = get_model_dir(opt)
    shutil.rmtree(model_dir) #ignore_errors=True
