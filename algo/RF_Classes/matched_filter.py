"""MatchedFilter Class.
This class creates Matched Filter for certain radar waveforms.
Supported waveforms:
    LFM (up and down)
    Barker

Created on Tue Apr  7 2020
Version: 1.0
@author: v025222357 Amir Sher
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from algo.RF_algo.get_radars_from_lib import generate_waveform_name


class MatchedFilter():
    """Create Matched Filter for certain radar waveforms."""
    def __init__(self, mod_type, fs, offset, pw, bw=0, chips='1', sweep_dir='up'):
        """Create new Matched Filter for certain radar waveforms.

        Parameters
        ----------
        mod_type : string
            Modulation type: 'LFM', 'Barker'.
        fs : float
            Sampling frequency.
        offset : float
            Frequency offset of generated Matched Filter.
        pw : float
            Pulse Width of matched filter (in seconds).
        bw : float, optional
            Bandwidth of matched filter (in Herz). The default is 0, though
            mandatory for LFM.
        chips : integer, optional
            Number of chips for Barker code. The default is '1', though
            mandatory for Barker.
        sweep_dir : string, optional
            Sweep direction for LFM: 'up'/'down'. The default is 'up'.

        Returns
        -------
        None.

        """
        self.mod_type = mod_type.upper()
        self.fs = fs
        self.ts = 1 / fs
        self.pw = pw
        self.bw = bw
        self.nsamples = int(round(self.pw * self.fs))
        self.t = np.arange(0, self.pw, self.ts)
        self.offset = offset
        self.chips = chips
        self.sweep_dir = sweep_dir
        self.signal_baseband = self.get_signal()
        self.signal_mod = self.signal_baseband if offset == 0 else self. \
            modulate_signal()
        self.coeff = self.get_coeff()
        self.auto_corr, self.mf_gain, self.mf_width = self.autocorr()
        self.mf_name = self.get_mf_name()

    def get_signal(self):
        """Generate baseband Matched Filter signal according to mod type.

        Returns
        -------
        sig : complex float array
            IQ presentation of baseband Matched Filter according to mod_type.
        """
        #Generate signal for each modulation type
        if self.mod_type == 'LFM':
            sig = self.get_signal_lfm()
        elif self.mod_type == 'BARKER':
            sig = self.get_signal_barker()
        else:
            sig = np.array([0])
        return sig

    def modulate_signal(self, sig=0, offset=0):
        """Modulate the Matched Filter signal to offset frequency.

        Parameters
        ----------
        sig : complex float array, optional
            The signal to be modulated. The default is 0 (self instance baseband
            MF).
        offset : float, optional
            Frequency offset of modulation. The default is 0 (no modulation).

        Returns
        -------
        complex float array
            IQ presentation of baseband Matched Filter according to mod_type.
        """
        if offset == 0:
            offset = self.offset
        if not isinstance(sig, np.ndarray):
            sig = self.signal_baseband
        return sig * np.exp(1j * 2 * np.pi * offset * self.t)

    def get_coeff(self):
        """Return Matched Filter coefficients.
        Coefficients are the conjugated and time reversed signal.

        Returns
        -------
        complex float array
            IQ presentation of Matched Filter coefficients.
        """
        c = np.conj(self.signal_mod)
        return c[::-1]

    def get_gain(self):
        """Return the gain of the Matched Filter.

        Returns
        -------
        float
            The gain of the Matched Filter.
        """
        return (self.coeff * np.conj(self.coeff)).real

    def autocorr(self):
        """Perform auto-correlation of Matched Filter to calculate MF gain and
        width.

        Returns
        -------
        abs_corr : float array
            Auto-Correlation array result.
        gain : float
            Matched Filter Gain.
        width : int
            half height auto-correlation width.
        """
        corr = signal.correlate(self.signal_baseband, self.signal_baseband, mode='same')
        gain = max(corr).real
        abs_corr = abs(100 * corr / gain)
        corr_top = np.where(abs_corr > 50)[0]
        width = corr_top[-1] - corr_top[0]
        return abs_corr, gain, width

    def xcorr(self, x, normalize=1, score=50, min_width_factor=10, showplot=False
              ):
        """Perform correlation of Matched Filter coefficients with the input
        signal.

        Parameters
        ----------
        x : complex float array
            input IQ signal to test correlation with Matched Filter.
        normalize : float, optional
            Normalization factor to correlation result. The default is 1.
        score : integer, optional
            Required minimum score to test for correlation test pass. The
            default is 50.
        min_width_factor : integer, optional
            Minimum required correlation width compared to optimum. The default
            is 10.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        corr : complex float array
            Correlation array result.
        corr_pass : Boolean
            True if correlation pass the input score.
        matched_scores : float array
            Array of correlation scores of each correlation points.
        matched_time : float array
            Array of time stamps of correlation points.
        """
        corr = 100 * signal.correlate(x, self.signal_mod, mode='same') / (self.
                                                                          mf_gain * normalize)
        abs_corr = abs(corr)
        t_axis = np.arange(len(x)) * self.ts
        corr_pass = False
        hits = 0
        matched_scores = []
        matched_time = -1

        if (score > 0) and (abs_corr > score).any():
            pw = int(self.pw * self.fs)
            peaks, peaks_dict = signal.find_peaks(abs_corr, height=score,
                                                  distance=pw,
                                                  rel_height=0.5, prominence=
                                                  score,
                                                  width=2)
            hits = len(peaks)
            if hits > 0:
                width_factor = np.mean(peaks_dict['widths']) / self.mf_width
                matched_scores = abs_corr[peaks]
                matched_time = peaks * self.ts
                corr_pass = True if width_factor < min_width_factor else False

        if showplot:
            fig, ax = plt.subplots()
            ax.set_title(f'Correlation with {self.mf_name} Matched Filter')
            if score > 0:
                if corr_pass:
                    passed = 'Passed.'
                else:
                    passed = 'Failed.'
                ax.set_title(f'{passed} ({hits}) Correlations with {self.mf_name} Matched Filter')
                score_line = plt.Line2D((0, t_axis[-1]),
                                        (score, score), ls='--', c='red')
                ax.add_line(score_line)
            if corr_pass:
                ax.plot(matched_time, matched_scores, 'rv')
            ax.plot(t_axis, abs_corr, 'b')
            ax.grid(True)
            plt.show()
        return corr, corr_pass, matched_scores, matched_time

    def mf_plot(self):
        """Plot Matched Filter coefficients and autocorellation.

        Returns
        -------
        None.
        """
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.subplots_adjust(hspace=0.6)
        plt.title(f'Real Coeffecients of {self.mf_name} Matched Filter')
        plt.stem(self.signal_mod.real, use_line_collection=True)

        plt.subplot(3, 1, 2)
        plt.title(f'Imag Coeffecients of {self.mf_name} Matched Filter')
        plt.stem(self.signal_mod.imag, use_line_collection=True)

        plt.subplot(3, 1, 3)
        plt.title(f'Autocorrelation of {self.mf_name} Matched Filter')
        plt.plot(self.auto_corr)
        plt.show()

    def get_mf_name(self):
        """Return Matched Filter name based on the mod_type and parameters.

        Returns
        -------
        name : string
            Match Filter name.
        """
        # Define Matched Filter name for plots and classification
        name = generate_waveform_name(self.mod_type, sweep_dir=self.sweep_dir,
                                      chips=self.chips)
        return name

    def get_signal_lfm(self):
        """Generate baseband LFM Matched Filter according to instance parameters.

        Returns
        -------
        sig : complex float array
            IQ presentation of baseband LFM matched filter.
        """
        #Each MatchedFilter sub class will implement signal generation
        if self.sweep_dir == 'up':
            sigr = signal.chirp(self.t, 0, self.pw, self.bw, 'linear',
                                phi=0)
            sigi = signal.chirp(self.t, 0, self.pw, self.bw, 'linear',
                                phi=-90)
        else:
            sigr = signal.chirp(self.t, self.bw, self.pw, 0,
                                'linear',
                                phi=0)
            sigi = signal.chirp(self.t, self.bw, self.pw, 0,
                                'linear',
                                phi=-90)
        sig = sigr + 1j * sigi
        return self.modulate_signal(sig, offset=-self.bw / 2)

    def get_signal_barker(self):
        """Generate baseband Barker Matched Filter according to instance
        parameters.

        Returns
        -------
        sig : complex float array
            IQ presentation of baseband Barker matched filter.
        """
        bc = {'1': (1)}
        bc.update({'2': (1, -1)})
        bc.update({'2b': (1, 1)})
        bc.update({'3': (1, 1, -1)})
        bc.update({'4': (1, 1, 1, -1)})
        bc.update({'4b': (1, 1, -1, 1)})
        bc.update({'5': (1, 1, 1, -1, 1)})
        bc.update({'7': (1, 1, 1, -1, -1, 1, -1)})
        bc.update({'11': (1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1)})
        bc.update({'13': (1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1)})

        barker = bc[self.chips]
        nchips = len(barker)
        chip_samples = int(round(self.nsamples / nchips))
        # modify nsamples and t according to the barker code
        self.nsamples = int(nchips * chip_samples)
        t = np.arange(0, self.nsamples * self.ts, self.ts)
        t_samples = min(len(t), self.nsamples)
        self.t = t[0:t_samples] # fix problem with some chips
        sig = []
        for i in range(0, nchips):
            b = np.ones((1, chip_samples), float) * barker[i]
            sig = np.concatenate((sig, b), axis=None)
        return sig


def main():
    lfm1 = MatchedFilter(mod_type="LFM", fs=31250000, offset=0, pw=1.28e-05, bw=
                         761452.24)
    lfm1.mf_plot()
    print(lfm1.mf_width)
    lfm1.xcorr(lfm1.signal_mod, score=50, showplot=True)

    lfm2 = MatchedFilter(mod_type="LFM", fs=1e6, offset=250000, pw=5e-4, bw=
                         10000)
    lfm2.mf_plot()
    print(lfm2.mf_width)
    lfm2.xcorr(lfm2.signal_mod, score=50, showplot=True)

    brk7 = MatchedFilter(mod_type="BARKER", fs=1e6, offset=0, pw=5e-4, chips='7')
    brk7.mf_plot()
    print(brk7.mf_width)
    brk7.xcorr(brk7.signal_mod, score=50, showplot=True)

    brk4b = MatchedFilter(mod_type="BARKER", fs=1e6, offset=0, pw=5e-4, chips=
                          '4b')
    brk4b.mf_plot()
    print(brk4b.mf_width)
    brk4b.xcorr(brk4b.signal_mod, score=50, showplot=True)


if __name__ == '__main__':
    main()