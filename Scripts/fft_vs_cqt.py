import argparse
import matplotlib.pyplot as plt
import librosa 
import numpy as np

def sampleSine(freq: float, sample_rate: float, length: float):
    t = np.arange(0, length, 1 / sample_rate)

    return np.sin(freq * 2 * np.pi * t)


def sampleComplexSound(freqs: list, sample_rate: float, t_start: float, length: float):
    # length in samples
    s_length = sample_rate * (t_start + length)
    signal = np.zeros((round(s_length), ))

    s_start = round(t_start * sample_rate)

    for freq in freqs:
        signal[s_start:] += sampleSine(freq, sample_rate, length)

    return signal


def sampleComplexSine(freq: float, no_harmonics: int, sample_rate: float, t_start: float, length: float, padto: float):
    """
    Sample a sound build of a chain of sinosindal harmonics.

    Parameters: 
        freq - fundamental frequency
        no_harmonics - number of harmonics that build the signal, starting with the fundamental frequency
        sample_rate - generated signal sample rate
        t_start - timestamp of the start of generated signal
        length - length of the generated signal in seconds
    """

    freq_harmonics = [freq * (1 + no) for no in range(no_harmonics)]

    signal = sampleComplexSound(freq_harmonics, sample_rate, t_start, length)

    s_full_length = round(padto * sample_rate)
    signal = np.pad(signal, (0, s_full_length - signal.size))

    return signal


def main(transform: str):
    freqs = [196, 392, 784]
    no_harmonics = 20
    t_start_list = 0.0, 0.5, 1 
    length = 0.5
    full_length = length * 3
    sample_rate = 44100

    signal_list = []

    for freq, t_start in zip(freqs, t_start_list):
        signal_list.append(
                sampleComplexSine(
                    freq=freq,
                    no_harmonics=no_harmonics,
                    sample_rate=sample_rate,
                    t_start=t_start,
                    length=length,
                    padto=full_length))

    signal = np.stack(signal_list)
    signal = np.sum(signal, axis=0)

    if transform == 'stft':
        S = np.abs(librosa.stft(signal))
        y_axis_type = 'fft'
    elif transform == 'cqt':
        S = np.abs(librosa.cqt(signal, 
                               sr=sample_rate, n_bins=24 * 9,
                               bins_per_octave=24))
        y_axis_type = 'cqt_hz'
    else:
        print(f"Transform {transform} not available, returning")
        return

    fig, ax = plt.subplots()

    img = librosa.display.specshow(
            # librosa.amplitude_to_db(S, ref=np.max),
            np.abs(S),
            sr=sample_rate,
            bins_per_octave=24,
            y_axis=y_axis_type, x_axis='time', ax=ax)

    ax.set_title(f"{transform.upper()} Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.show()

    # show the graph until dismissed
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare FFT output against CQT")
    parser.add_argument(
            'transform',
            default='stft',
            type=str,
            choices=['stft', 'cqt'],
            help="Transform type")

    args = parser.parse_args()
    main(**vars(args))
