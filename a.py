import mat4py
import mne
from gdflib import GdfEntries, Node
import mne
from mne import io
from mne.connectivity import spectral_connectivity, seed_target_indices
from mne.datasets import sample
from mne.time_frequency import AverageTFR
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

student1 = mat4py.loadmat('Desktop/test/B0101T.mat')
#student1 = student1['student']
print(student1)
# print type(student1)  # dict
# print ','.join(['%s' % key for key, val in student1.iteritems()])  # age,score,name,sex


#
# entities = GdfEntries()
# entities.add_node(Node(name='B0101T', label='This is the first node'))
# entities.add_node(Node(name='B0102T', label='This is the second node'))
# entities.link('node1', 'node2')
# print entities.dumps()

raw = mne.io.read_raw_edf('Desktop/test/B0101T.gdf', stim_channel=None,preload=True)  # load data
print(raw._raw_extras[0])
print(raw.info)
raw.save("sample1_raw.fif")
raw_fname="sample1_raw.fif"
raw = mne.io.read_raw_fif(raw_fname)
print(raw.find_edf_events)
raw.info['bads'] = ['EEG c3']  # mark bad channels
raw.filter(l_freq=None, h_freq=40.0)  # low-pass filter
events = mne.find_events(raw,stim_channel=None)  # extract events and epoch data
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()  # compute evoked
evoked.plot()  # butterfly plot the evoked data
cov = mne.compute_covariance(epochs, tmax=0, method='shrunk')
fwd = mne.read_forward_solution(fwd_fname, surf_ori=True)
inv = mne.minimum_norm.make_inverse_operator(  raw.info, fwd, cov, loose=0.2)  # compute inverse operator
stc = mne.minimum_norm.apply_inverse(  evoked, inv, lambda2=1. / 9., method='dSPM')  # apply it
stc_fs = stc.morph('fsaverage')  # morph to fsaverage
stc_fs.plot()  # plot source data on fsaverage's brain