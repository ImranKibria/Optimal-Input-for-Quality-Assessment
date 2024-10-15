import array
import struct
import webrtcvad                                # Py-WebRTC VAD package is a port to the WebRTC project by Google. 

vad = webrtcvad.Vad()  
vad.set_mode(1)                                 # 0-3 are the modes, 0 being the least aggressive.

def find_vad(fr_data, sr, fr_dur=30):
    # frame/window duration is 10ms because we have 512 samples at 48kHz fs 

    tot_samples = int(sr * fr_dur / 1000)       # 480 samples
    raw_floats = fr_data[0:tot_samples]

    floats = array.array('f', raw_floats)

    samples = [int(max(min(sample, 1), -1) * 32767) for sample in floats]    # max(min(sample, 1), -1) for clipping in 16kHz downsampled data

    raw_ints = struct.pack("<%dh" % len(samples), *samples)

    return vad.is_speech(raw_ints, sr)



# # THE FOLLOWING IS THE ORIGINAL VERSION OF VOICE ACTIVITY DETECTION (link: https://pypi.org/project/webrtcvad-wheels/)
# 
# import webrtcvad
# vad = webrtcvad.Vad() 

# vad.set_mode(1)

# # Run the VAD on 10 ms of silence. The result should be False.
# sample_rate = 16000
# frame_duration = 10  # ms
# frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
# print ('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))
