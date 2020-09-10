import skimage.io as io
from pathlib import Path
import seaborn as sns
from scipy.spatial import distance
import numpy as np
from sklearn.metrics import recall_score
from utils import metrics, metrics
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import Pool
def eval_img(gt_file, pred_file, img_name, uncertainty_file=None):
    gt_img = io.imread(gt_file, as_gray=True)
    gt = (gt_img > 200).astype(int)
    pred_img = io.imread(pred_file, as_gray=True)
    pred = (pred_img > 200).astype(int)
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    scores = {}
    scores['image'] = img_name
    scores['dice']= metrics.dice_coefficient(gt_flat, pred_flat)
    scores['euclidian'] = distance.euclidean(gt_flat, pred_flat)
    scores['IOU'] = metrics.IOU(gt_flat, pred_flat)
    scores['specificity'] = metrics.specificity(gt_flat, pred_flat)
    scores['sensitivity'] = recall_score(gt_flat, pred_flat)
    scores['line_accuracy'] = metrics.line_accuracy(gt_flat, pred_flat)
    if uncertainty_file:
        uncertainty_img =  io.imread(uncertainty_file, as_gray=True)
        uncertainty = uncertainty_img / 65535
        scores['uncertainty'] = uncertainty.mean()
    return scores


def evaluate(gt_path, pred_path, out_path=None):
    pred_path = Path(pred_path)
    gt_path = Path(gt_path)
    if not out_path:
        out_path = pred_path

    pred_files =[]
    gt_files = []
    img_names = []
    uncertainty_files = []
    for f in gt_path.glob("*.png"):
        gt_files.append(str(f))
        if "_zones.png" in f.name:
            basename = f.name[:f.name.rfind('_')]
        else:
            basename = f.stem
        img_names.append(basename + '.png')
        if Path(pred_path, basename + "_pred.png").exists():
            pred_files.append(str(Path(pred_path, basename + '_pred.png')))
        elif Path(pred_path, basename + ".png").exists():
            pred_files.append(str(Path(pred_path, basename + '.png')))
        else:
            raise FileNotFoundError(str(Path(pred_path, basename + '.png')) + " not found")
        if Path(pred_path, basename + "_uncertainty.png").exists():
            uncertainty_files.append(str(Path(pred_path, basename + '_uncertainty.png')))


    if len(pred_files) != len(gt_files) != len(img_names):
        raise AssertionError("Prediction and Ground truth set size does not match")

    if len(uncertainty_files) != 0 and len(uncertainty_files) != len(pred_files):
        raise AssertionError("Nr of Uncertainty images does not match Nr of Prediction images")


    p = Pool()
    if len(uncertainty_files) > 0:
        set = zip(gt_files, pred_files, img_names, uncertainty_files)
    else:
        set = zip(gt_files, pred_files, img_names)
    scores = p.starmap(eval_img, set)
    scores = pd.DataFrame(scores)
    scores.to_pickle(Path(out_path, 'scores.pkl'))

    # Create summary report
    header ='Dice\tIOU\tEucl\tSensitivity\tSpecificitiy'
    report = (str(np.mean(scores['dice'])) + '\t'
            + str(np.mean(scores['IOU'])) + '\t'
            + str(np.mean(scores['euclidian'])) + '\t'
            + str(np.mean(scores['sensitivity'])) + '\t'
            + str(np.mean(scores['specificity'])) + '\n')
    print(header)
    print(report)

    with open(str(Path(out_path, 'ReportOnModel.txt')), 'w') as f:
        f.write(header + '\n')
        f.write(report)

    return scores



#def eval(gt_file, pred_file,img_name, uncertainty_file=None):
#    gt_img = tf.image.decode_png(gt_file)
#    gt = tf.cast(gt_img > 200, tf.float32)
#    pred_img = tf.image.decode_png(pred_file)
#    pred = tf.cast(pred_img > 200,tf.float32)
#    pred_flat = tf.reshape(pred, [-1])
#    gt_flat = tf.reshape(gt, [-1])
#    scores = {}
#    scores['image'] = img_name
#    scores['dice']= metrics.dice_coefficient_tf(gt_flat, pred_flat)
#    scores['euclidian'] = metrics.euclidian_tf(gt_flat, pred_flat)
#    scores['IOU'] = metrics.IOU_tf(gt_flat, pred_flat)
#    scores['specificity'] = metrics.specificity_tf(gt_flat, pred_flat)
#    if uncertainty_file:
#        uncertainty_img =  tf.image.decode_png(uncertainty_file)
#        uncertainty = uncertainty_img / 65535
#    else:
#        uncertainty = 0.
#    scores['uncertainty'] = tf.reduce_mean(uncertainty)
#    m = tf.metrics.Recall()
#    m.update_state(gt_flat, pred_flat)
#    scores['sensitivity'] = m.result()
#    return scores
#
#def evaluate_tf(gt_path, pred_path, batch_size=4):
#    pred_path = Path(pred_path)
#    gt_path = Path(gt_path)
#
#    pred_files =[]
#    gt_files = []
#    img_names = []
#    uncertainty_files = []
#    for f in gt_path.glob("*.png"):
#        gt_files.append(str(f))
#        if "_zones.png" in f.name:
#            basename = f[:f.name.rfind('_')]
#        else:
#            basename = f.stem
#        img_names.append(basename + '.png')
#        if Path(pred_path, basename + "_pred.png").exists():
#            pred_files.append(str(Path(pred_path, basename + '_pred.png')))
#        elif Path(pred_path, basename + ".png").exists():
#            pred_files.append(str(Path(pred_path, basename + '.png')))
#        else:
#            raise FileNotFoundError(str(Path(pred_path, basename + '.png')) + " not found")
#        if Path(pred_path, basename + "_uncertainty.png").exists():
#            uncertainty_files.append(str(Path(pred_path, basename + '_uncertainty.png')))
#
#
#    if len(pred_files) != len(gt_files) != len(img_names):
#        raise AssertionError("Prediction and Ground truth set size does not match")
#
#    if len(uncertainty_files) != 0 and len(uncertainty_files) != len(pred_files):
#        raise AssertionError("Nr of Uncertainty images does not match Nr of Prediction images")
#
#    pred_set = tf.data.Dataset.from_tensor_slices(pred_files)
#    gt_set = tf.data.Dataset.from_tensor_slices(gt_files)
#    name_set = tf.data.Dataset.from_tensor_slices(img_names)
#    uncertainty_set = tf.data.Dataset.from_tensor_slices(uncertainty_files)
#
#    if len(uncertainty_files) > 0:
#        set = tf.data.Dataset.zip((gt_set, pred_set,name_set, uncertainty_set))
#    else:
#        set = tf.data.Dataset.zip((gt_set, pred_set,name_set))
#
#    scores = set.map(eval)
#    scores_list = scores.reduce([], lambda l, e: l.append(e))
#    scores = pd.DataFrame(scores_list)
#
#
#    print("BLA")





#def evaluate_dice_only(imgs, gt_imgs, predictions):
#    dice = []
#    for img, gt_img, pred in zip(imgs, gt_imgs, predictions):
#        gt = (gt_img > 200).astype(int)
#        gt_flat = gt.flatten()
#        pred_flat = pred.flatten()
#        dice.append(metrics.dice_coefficient(gt_flat, pred_flat))
#    return dice


#def evaluate_dice_only_(img_path, gt_path, prediction_path):
#    dice = []
#    for filename in Path(img_path).rglob('*.png'):
#        gt_img = io.imread(Path(gt_path, filename.stem + '_zones.png'), as_gray=True)
#        if (Path(prediction_path, filename.stem + '_pred.png')).exists():
#            pred_img = io.imread(Path(prediction_path, filename.stem + '_pred.png'), as_gray=True)
#        elif Path(prediction_path,filename.name).exists():
#            pred_img = io.imread(Path(prediction_path,filename.name), as_gray=True) # Legacy before predictions got pred indentifier
#        else:
#            print(str(Path(prediction_path,filename.name)) + " not found")
#            continue
#
#
#        gt = (gt_img > 200).astype(int)
#        pred = (pred_img > 200).astype(int)
#
#        gt_flat = gt.flatten()
#        pred_flat = pred.flatten()
#        print(filename.name)
#        dice.append(metrics.dice_coefficient(gt_flat, pred_flat))
#    return dice

def plot_history(history, out_file, xlim=None, ylim=None, title=None):
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot( history['loss'], 'X-', label='training loss', linewidth=4.0)
    plt.plot(history['val_loss'], 'o-', label='val loss', linewidth=4.0)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if title:
        plt.title(title)
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--')
    plt.savefig(out_file, bbox_inches='tight', format='png', dpi=200)
    plt.show()

def eval_uncertainty(file, out_file, vmin=0, vmax=0.2):
    uncertainty = np.load(file)
    plt.figure()
    sns.heatmap(uncertainty, vmin=vmin, vmax=vmax, xticklabels=False, yticklabels=False)
    plt.savefig(out_file, bbox_inches='tight', format='png', dpi=200)

if __name__ == '__main__':
    path = Path('/home/andreas/glacier-front-detection/tmp')

    test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn/val')
    evaluate(Path(test_path, 'masks'), path)
    #test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/test')
    #history = pickle.load(open(next(path.glob('history*.pkl')), 'rb'))
    #history = pd.read_csv(Path(path,'model_1_history.csv'))
    #plot_history(history, Path(path, 'loss_plot.png') , xlim=(-10,30), ylim=(0,0.75), title='Set1')

    #evaluate(Path(test_path,'images'), Path(test_path,'masks'), path)

    #evaluate(Path(test_path, 'images'), Path(test_path, 'masks'), path)
    #test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/test')
    #evaluate(Path(test_path, 'images'), Path(test_path, 'masks'), '/home/andreas/glacier-front-detection/output_pix2pix_/output_Jakobshavn_pix2pix')
    #for d in Path('/home/andreas/glacier-front-detection/output_pix2pix_front_only').iterdir():
    #    if d.is_dir():
    #        test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/val/patches')
    #        evaluate(Path(test_path, 'masks'), d)
            #history = pickle.load(open(next(d.glob('history*.pkl')), 'rb'))
            #plot_history(history, Path(d, 'loss_plot.png')) # , xlim=(-10,130), ylim=(0,0.8))
