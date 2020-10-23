from DeepJetCore.TrainData import TrainData
from DeepJetCore.dataPipeline import TrainDataGenerator
from LayersRagged import RaggedConstructTensor
import index_dicts
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

td=TrainData()
td.readFromFile('/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/50_part_with_noise_Jul2020/converted/HGCalML_data/50_part_with_noise_Jul2020/988_windowntup.djctd')
gen = TrainDataGenerator()
gen.setBatchSize(100000)
gen.setSkipTooLargeBatches(False)
gen.setBuffer(td)

with tf.device('/CPU:0'):
    ragged_constructor = RaggedConstructTensor()

while True:
    feat, truth = next(gen.feedNumpyData())  # this is  [ [features],[truth],[None] ]

    if gen.lastBatch():
        break

    row_splits = feat[1][:, 0]

    with tf.device('/CPU:0'):
        feat, _ = ragged_constructor((feat[0], row_splits))
        truth, row_splits = ragged_constructor((truth[0], row_splits))

    for i in range(len(row_splits) - 1):
        with tf.device('/CPU:0'):
            features_s = feat[row_splits[i]:row_splits[i + 1]].numpy()
            truth_s = truth[row_splits[i]:row_splits[i + 1]].numpy()
            feature_dict = index_dicts.create_feature_dict(features_s) # follow this
            truth_dict = index_dicts.create_truth_dict(truth_s) # and this function to know all the keys

    print(feature_dict['recHitEnergy'].shape) # rechit energies
    print(feature_dict['recHitEta'].shape) # rechit eta
    print(feature_dict['recHitRelPhi'].shape) # rechit phi
    print(feature_dict['recHitX'].shape) # rechit X
    print(feature_dict['recHitY'].shape) # rechit Y
    print(feature_dict['recHitZ'].shape) # rechit Z
    print(feature_dict['recHitTime'].shape) # rechit Z

    print(truth_dict['truthHitAssignementIdx'].shape) # shower id
    print(truth_dict['truthRealEnergy'].shape) # energy of the shower

    print("Going to next window\n\n")
