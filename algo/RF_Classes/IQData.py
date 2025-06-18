"""IQData Class.
Read IQ samples data from xdat file and perform some signal processing on it.
Created on Tue Apr  7 2020
Version: 2.0
Updated on Sun May 31 2020
@author: v025222357 Amir Sher
"""
import numpy as np
import scipy.signal as signal
import pandas as pd
import math
import os
from os import path
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from utils import xhdr_utils as xhdr_utils
from utils import num_utils as nu


MIN_PROMINENCE = 7 # minimum prominence for detected signal


class IQData():
    """Read IQ data from xdat file and perform some signal processing on the
    IQ.
    """
    def __init__(self, iq_source, start_time=0, sample_length=0):
        """Create new IQData instance from xdat file or from another IQData
        object.

        Parameters
        ----------
        iq_source : string or IQData class
            string: Path of an xdat file to read IQ data from.
            if it is IQData class, the new class will get the IQ from it.
        start_time : float, optional
            start reading IQ data from start time (seconds).
            The default is 0, which is the beginning of the file.
        sample_length : float, optional
            Length of the IQ to read (in seconds).
            The default is 0, which is until the end of the file.

        Returns
        -------
        None.

        """
        if isinstance(iq_source, IQData): # Copy object to new instance
            self.iq_path = iq_source.iq_path
            self.xdat_ext = iq_source.xdat_ext
            self.xdat_name = iq_source.xdat_name
            self.xdat_name_ext = iq_source.xdat_name_ext
            self.xhdr_file = iq_source.xhdr_file
            self.file_path = iq_source.file_path
            self.acq_scale_factor = iq_source.acq_scale_factor
            self.fc = iq_source.fc
            self.fs = iq_source.fs
            self.num_samples = iq_source.num_samples
            self.span = iq_source.span
            self.start_capture = iq_source.start_capture
            self.start_time = iq_source.start_time
            self._define_period(start_time, sample_length)
            self.iq = iq_source.iq[self.start_sample:self.end_sample]
            self.signals = {}

        elif self._path_exists(iq_source): # Fetch metadata from xhdr and IQ from xdat files
            self.get_xhdr()
            self._define_period(start_time, sample_length)
            self.get_xdat()
            self.signals = {}
        else:
            print(f'Could not find the IQ file in {iq_source}')

    def get_xhdr(self):
        """Get metadata from xhdr file.
        Update the class parameters accordingly.
        Returns
        -------
        None.

        """
        acq_scale_factor, result = xhdr_utils.get_acq_scale_factor(self.xhdr_file)
        sample_rate, result = xhdr_utils.get_sample_rate(self.xhdr_file)
        center_frequency, result = xhdr_utils.get_center_frequency(self.xhdr_file)
        samples, result = xhdr_utils.get_samples(self.xhdr_file)

        self.fc = float(center_frequency) # center_frequency
        self.acq_scale_factor = float(acq_scale_factor)
        self.fs = float(sample_rate) # sample_rate
        self.num_samples = int(float(samples))
        acquisition_bandwidth, result = xhdr_utils.get_acquisition_bandwidth(self.xhdr_file)
        self.span = float(acquisition_bandwidth)
        if result is not None:
            self.span = self.fs

        #Parse date/time from 'start_capture' metadata
        start_capture_str, result = xhdr_utils.get_start_capture(self.xhdr_file)
        (y, j, h, m, s) = start_capture_str.split(':')
        (s1, s2) = s.split('.')
        s = s1 + '.' + s2[0:6]
        dt_string = y + ':' + (str(int(j))) + ':' + h + ':' + m + ':' + s
        dt_object = datetime.strptime(dt_string, "%Y:%j:%H:%M:%S.%f")
        self.start_capture = dt_object

    def get_xdat(self):
        """Get IQ data from xdat file and return the complex representation.
        Returns
        -------
        iq : Complex float array
            Complex IQ data from xdat file.
        """
        self.iq = xhdr_utils.get_complex_array(self.acq_scale_factor, self.ts,
                                               self.iq_path,
                                               self.num_samples,
                                               self.ts,
                                               self.start_sample,
                                               self.ts)
        return self.iq

    def save_xdat(self, file):
        """Save IQ data and metadata to new xdat and xhdr files.

        Parameters
        ----------
        file : string
            Path and file name.

        Returns
        -------
        None.

        """
        xhdr_utils.save_xhdr(file, self.acq_scale_factor, self.fc, self.fs,
                             self.span, self.start_capture, self.num_samples)
        xhdr_utils.save_xdat(file, self.iq, self.acq_scale_factor)

    def get_abs(self):
        """Get the absolute array of the complex IQ data.

        Returns
        -------
        iq_abs : float array
            Absolute array of IQ data.

        """
        if hasattr(self, 'abs'):
            return self.abs
        else:
            self.abs = np.abs(self.iq)
        return self.abs

    def _define_period(self, start_time=0, sample_length=0):
        """Define the samples and time variables according to the specified IQ
        start time and length.

        Parameters
        ----------
        start_time : float, optional
            Define the beginning of the IQ data (in seconds).
            The default is 0, which is the beginning of the file.
        sample_length : float, optional
            Define the length of the IQ to read (in seconds).
            The default is 0, which is until the end of the file.

        Returns
        -------
        None.

        """
        sample_length = 0 if sample_length < 0 else sample_length
        self.start_sample = int(math.floor(start_time * self.fs))
        self.start_time = start_time
        if self.start_sample < 1 or self.start_sample >= self.num_samples:
            self.start_sample = 0
            self.start_time = 0
        else:
            self.start_capture = self.start_capture + timedelta(seconds=
                                                                start_time)

        self.end_sample = int(math.ceil(self.start_sample +
                                        sample_length * self.fs))
        if self.end_sample > self.num_samples or sample_length == 0:
            self.end_sample = self.num_samples
        self.num_samples = self.end_sample - self.start_sample
        # Get time array according to sample rate and number of data points
        self.ts = 1 / self.fs
        self.t = np.arange(0, self.num_samples, 1) * self.ts

    def _path_exists(self, iq_path):
        """Check if iq path exists as a file and fetch xhdr file path.

        Parameters
        ----------
        iq_path : string
            the full path and name of the xdat file.

        Returns
        -------
        bool
            True if a file exists in iq_path.
            False if otherwise.
        """
        self.iq_path = ""
        if isinstance(iq_path, str):
            if (iq_path[-5:] == '.xdat') or (iq_path[-5:] == '.xhdr'):
                name = iq_path[:-5]
            else:
                name = iq_path
            xdat_file = name + '.xdat'
            xhdr_file = name + '.xhdr'
            if path.exists(xdat_file) and path.exists(xhdr_file):
                #Parse path name
                self.iq_path = xdat_file
                (self.file_path, self.xdat_name_ext) = os.path.split(self.iq_path)
                self.xdat_name = self.xdat_name_ext[0:-5]
                self.xdat_ext = 'xdat'
                self.xhdr_file = xhdr_file
                return True
            else:
                return False
        else:
            return False

    def copy(self, start_time=0, sample_length=0):
        """Duplicate the IQData instance and cut the IQ signal.

        Parameters
        ----------
        start_time : float, optional
            Define the beginning of the IQ data (in seconds).
            The default is 0, which is the beginning of the file.
        sample_length : float, optional
            Define the length of the IQ to read (in seconds).
            The default is 0, which is until the end of the file.

        Returns
        -------
        newobject : IQData
            A new IQData instance which is a trimmed copy of current instance.

        """
        newobject = IQData(self, start_time, sample_length)
        return newobject

    def cut_iq(self, start_time=0, sample_length=0):
        """Cut (shorten) the IQ array.

        Parameters
        ----------
        start_time : float, optional
            Define the beginning of the IQ data (in seconds).
            The default is 0, which is the beginning of the file.
        sample_length : float, optional
            Define the length of the IQ to read (in seconds).
            The default is 0, which is until the end of the file.

        Returns
        -------
        None.

        """
        self._define_period(start_time, sample_length)
        self.iq = self.iq[self.start_sample:self.end_sample]
        if hasattr(self, 'f_range'):
            del self.f_range
            del self.psd
        if hasattr(self, 'abs'):
            self.abs = self.abs[self.start_sample:self.end_sample]

    def new_iq(self, iqdata, start_time=0, sample_length=0):
        """Put new IQ values (with same sample rate) in the object.
        Use this method to apply changes to IQ, for example: lowpass filter.
        Parameters
        ----------
        iqdata : complex float array
            New IQ data.
        start_time : float, optional
            Define the beginning of the IQ data (in seconds).
            The default is 0, which is the beginning of the file.
        sample_length : float, optional
            Define the length of the IQ to read (in seconds).
            The default is 0, which is until the end of the file.

        Returns
        -------
        None.

        """
        self._define_period(start_time, sample_length)
        self.iq = iqdata[self.start_sample:self.end_sample]
        if hasattr(self, 'f_range'):
            del self.f_range
            del self.psd
        if hasattr(self, 'abs'):
            del self.abs

    def modulate(self, offset, showplot=False):
        """Modulate the IQ signal to offset frequency.
        This method returns a new IQ but doesn't change self.iq

        Parameters
        ----------
        offset : float
            The frequency to modulate the self.iq signal.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        iqmod : complex float array
            The modulated IQ signal.
        """
        iqmod = self.iq * np.exp(1j * 2 * np.pi * offset * self.t)
        if showplot:
            NFFT = 513
            #pwelch before modulate
            f_range, psd = signal.welch(self.iq, self.fs, window=('kaiser', 7),
                                        nfft=NFFT * 2, return_onesided=False,
                                        detrend=False)
            psd = np.roll(10 * np.log10(psd), NFFT)
            #pwelch after modulate
            f_range, psd_mod = signal.welch(iqmod, self.fs, window=('kaiser',
                                                                    7),
                                            nfft=NFFT * 2, return_onesided=False,
                                            detrend=False)
            psd_mod = np.roll(10 * np.log10(psd_mod), NFFT)
            f_range = np.roll(f_range, NFFT)
            plt.figure()
            plt.plot(f_range, psd, 'b', label='Original')
            plt.plot(f_range, psd_mod, 'g', label='Modulated')
            off_num = nu.format_num(offset)
            plt.title(f'Signal Modulated by {off_num} Hz')
            plt.xlabel('Freq')
            plt.grid()
            plt.legend()
            plt.show()
        return iqmod

    def pwelch(self, n=513, showplot=False):
        """Calculate Power Spectral Density using Welch function.

        Parameters
        ----------
        n : integer, optional
            Half length of FFT. The default is 513.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        f_range : float array
            Array of frequencies, the frequency range of the PSD result.
        psd : float array
            Power spectral density of self.iq.
        """
        if n / 2 == 0:
            n = n + 1
        nft = 2 * n
        f_range, p = signal.welch(self.iq, self.fs, window=('kaiser', 3),
                                  nfft=nft, return_onesided=False, detrend=False)
        psd = 10 * np.log10(p)

        #Roll the vectors so they start with negative freq values
        self.f_range = np.roll(f_range, n)
        self.psd = np.roll(psd, n)
        self.psd = self.psd[abs(self.f_range) < self.span / 2]
        self.psd_noise_level = min(self.psd)
        self.f_range = self.f_range[abs(self.f_range) < self.span / 2]
        if showplot:
            plt.figure()
            plt.plot(self.f_range, self.psd, 'b')
            plt.title('Welch Power Spectrum Density')
            plt.xlabel('Freq (Hz)')
            plt.ylabel('PSD log (Power/Hz)')
            plt.grid()
            plt.show()
        return self.f_range, self.psd

    def butter_low_pass(self, f_cutoff, order=10, margin=0, showplot=False):
        """Perform lowpass filter on IQ signal.
        This method returns the result but doesn't change self.iq

        Parameters
        ----------
        f_cutoff : float
            Cutoff frequency for lowPass filter.
        order : int, optional
            The order of the butterworth filter. The default is 10.
        margin : int, optional
            Add margin to the cutoff, as percentage of the cutoff. The default
            is 0.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        sig_filt : complex float array
            IQ vector.
        """
        nyq = 0.5 * self.fs
        f_cutoff = f_cutoff * (1 + margin / 100)
        cutoff = f_cutoff / nyq
        cutoff = 1 if cutoff > 1 else cutoff
        sos = signal.butter(order, cutoff, btype='lowpass', output='sos')
        sig_filt = signal.sosfilt(sos, self.iq)
        if showplot:
            NFFT = 513
            #pwelch before lowpass
            f_range, psd = signal.welch(self.iq, self.fs, window=('kaiser', 7),
                                        nfft=NFFT * 2, return_onesided=False,
                                        detrend=False)
            psd = np.roll(10 * np.log10(psd), NFFT)
            #pwelch after lowpass
            f_range, psd_filt = signal.welch(sig_filt, self.fs, window=('kaiser',
                                                                        7),
                                             nfft=NFFT * 2, return_onesided=False,
                                             detrend=False)
            psd_filt = np.roll(10 * np.log10(psd_filt), NFFT)
            f_range = np.roll(f_range, NFFT)
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(f_range, psd, 'b', label='Original')
            plt.plot(f_range, psd_filt, 'g', label='Filtered')
            cutoff_num = nu.format_num(f_cutoff)
            plt.title(f'LowPass Filter ({cutoff_num}) Hz')
            plt.xlabel('Freq (Hz)')
            plt.grid()
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(self.t, self.iq.real, 'b', label='Original')
            plt.plot(self.t, sig_filt.real, 'g', label='Filtered')
            plt.xlabel('Time [s]')
            plt.grid()
            plt.tight_layout(h_pad=1.0)
            plt.show()
        return sig_filt

    def butter_band_pass(self, flo, fhi, order=10, showplot=False):
        """Perform bandpass filter on IQ signal.
        This method returns the result but doesn't change self.iq

        Parameters
        ----------
        flo : float
            Lower cutoff frequency for bandpass filter.
        fhi : float
            Higher cutoff frequency for bandpass filter.
        order : int, optional
            The order of the butterworth filter. The default is 10.
        margin : int, optional
            Add margin to the cutoff, as percentage of the cutoff. The default
            is 0.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        sig_filt : complex float array
            IQ vector.
        """
        #Perform bandpass filter on IQ signal
        # This method returns the result but doesn't change self.iq
        nyq = 0.5 * self.fs
        low = flo / nyq
        high = fhi / nyq
        sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
        sig_filt = signal.sosfilt(sos, self.iq)
        if showplot:
            NFFT = 513
            #pwelch before bandpass
            f_range, psd = signal.welch(self.iq, self.fs, window=('kaiser', 7),
                                        nfft=NFFT * 2, return_onesided=False,
                                        detrend=False)
            psd = np.roll(10 * np.log10(psd), NFFT)
            #pwelch after bandpass
            f_range, psd_filt = signal.welch(sig_filt, self.fs, window=('kaiser',
                                                                        7),
                                             nfft=NFFT * 2, return_onesided=False,
                                             detrend=False)
            psd_filt = np.roll(10 * np.log10(psd_filt), NFFT)
            f_range = np.roll(f_range, NFFT)
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(f_range, psd, 'b', label='Original')
            plt.plot(f_range, psd_filt, 'g', label='Filtered')
            flo_n = nu.format_num(flo)
            fhi_n = nu.format_num(fhi)
            plt.title(f'BandPass Filter ({flo_n})-({fhi_n}) Hz')
            plt.xlabel('Freq [Hz]')
            plt.grid()
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(self.t, self.iq.real, 'b', label='Original')
            plt.plot(self.t, sig_filt.real, 'g', label='Filtered')
            plt.xlabel('Time [s]')
            plt.grid()
            plt.tight_layout(h_pad=1.0)
            plt.show()
        #w, h = signal.sosfreqz(sos, worN=2000)
        return sig_filt

    def get_freq_peaks(self, min_distance, min_width, min_prominence, showplot=
                       False):
        """Analyze the IQ spectral domain (welch psd) and search for peaks and
        valleys.

        Parameters
        ----------
        min_distance : float
            Minimum distance between neighbour peaks.
        min_width : float
            Minimum width of selected peaks.
        min_prominence : float
            Minimum prominence of selected peaks.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        peaks : float array
            Array of the peaks power density.
        peaks_i : integer array
            Array of indexes of the peaks in the PSD array.
        peaks_f : float array
            The frequency offset of each peak.
        valleys_f : float array
            The frequency offset of valleys between peaks.
        prominences : float array
            The prominences of each peak.
        """
        #make sure there is welch data
        if not hasattr(self, 'f_range'):
            self.pwelch()

        # Find all peaks and valleys in spectrogram
        peaks, peaks_dict = signal.find_peaks(self.psd, prominence=min_prominence,
                                              distance=min_distance, width=
                                              min_width)
        valleys_dict = dict.fromkeys(peaks_dict['left_bases'])
        valleys_dict.update(dict.fromkeys(peaks_dict['right_bases']))
        valleys = list(valleys_dict)
        valleys = np.sort(valleys)

        prominences = peaks_dict['prominences']
        valleys_f = self.f_range[valleys] if len(valleys) > 0 else []
        peaks_i = peaks # index of peaks in psd vector
        peaks_f = self.f_range[peaks_i] if len(peaks_i) > 0 else () # frequency
        #of peaks
        peaks = self.psd[peaks_i] if len(peaks_i) > 0 else () # level values of
        #peaks

        if showplot:
            fig, ax = plt.subplots()
            plt.xlabel('Freq [Hz]')
            plt.ylabel('PSD log (Power/Hz)')
            plt.grid()
            ax.plot(self.f_range, self.psd, 'b')
            ax.plot(peaks_f, peaks, 'rx')
            valleys_v = self.psd[valleys] if len(valleys) > 0 else []
            ax.plot(valleys_f, valleys_v, 'g*')
            ax.set_title('Peaks and Valleys in Power Spectral Density')
            plt.show()
        return peaks, peaks_i, peaks_f, valleys_f, prominences

    def get_freq_signals(self, distance_percent=0.01, showplot=False):
        """Analyze the IQ spectral domain (welch psd) and search for signals.
        Analyze each peak for peak type (standalone or peak with side-lobes),
        bandwidth, and offset frequency.

        Parameters
        ----------
        distance_percent : float, optional
            Minimum distance between neighbour peaks. The default is 0.01.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        signals : dictionary
            Dictionary of identified peaks.
            signals data include:
                "bw": bandwidth,
                "flo": low frequency
                "fhi": high frequency
                "has_sll": Boolean, identify peak type
                "offset": frequency offset (in baseband)
                "cf": center frequency (in original RF band)
        """
        #make sure there is welch data
        if not hasattr(self, 'f_range'):
            self.pwelch()

        # Set minimum peak search resolution as distance percent of f_range
        min_distance = round(distance_percent * len(self.f_range))
        min_width = np.floor(min_distance / 2)
        min_prominence = 2

        #Find all peaks and valleys in spectrogram
        peaks, peaks_i, peaks_f, valleys_f, prominences = \
            self.get_freq_peaks(min_distance, min_width, min_prominence)

        ### Identify signals and register their frequency data.
        ### Locate signals among the detected peaks
        ### Identify side lobes and remove them from the peaks list ###

        tmp_peaks = peaks
        tmp_peaks_f = peaks_f
        tmp_prom = prominences
        tmp_peaks_i = peaks_i
        num_signals = 0
        self.signals = {} # start new signals dictionary
        while len(tmp_peaks) > 0 and (tmp_prom >= MIN_PROMINENCE).any():
            remove_peaks, bandwidth, flo, fhi = \
                self._find_lobes(tmp_peaks, tmp_peaks_i, tmp_peaks_f,
                                 valleys_f, tmp_prom)
            num_signals += 1
            if bandwidth > 0:
                flo = nu.num_round(flo, 10000)
                fhi = nu.num_round(fhi, 10000)
                bandwidth = fhi - flo
                offset = flo + bandwidth / 2
                signal_dict = {"bw": bandwidth,
                               "flo": flo,
                               "fhi": fhi,
                               "offset": nu.num_round(offset, '1M'),
                               "cf": self.fc}
                self.signals.update({num_signals: signal_dict})
            else:
                num_signals -= 1

            # remove detected sll's and registered main lobe from peaks array
            tmp_peaks = tmp_peaks[~remove_peaks]
            tmp_peaks_f = tmp_peaks_f[~remove_peaks]
            tmp_prom = tmp_prom[~remove_peaks]
            tmp_peaks_i = tmp_peaks_i[~remove_peaks]
        if showplot:
            fig, ax = plt.subplots()
            plt.xlabel('Freq [Hz]')
            plt.ylabel('PSD [dB/Hz]')
            plt.grid()
            ax.plot(self.f_range, self.psd, 'b')
            ax.plot(peaks_f, peaks, 'rx')
            valleys_i = [list(self.f_range).index(i) for i in valleys_f]
            valleys_v = self.psd[valleys_i] if len(valleys_i) > 0 else []
            ax.plot(valleys_f, valleys_v, 'g')
            min_graph = min(self.psd)
            max_graph = max(self.psd)
            for i in self.signals:
                rf_signal = self.signals[i]
                line_l = plt.Line2D((rf_signal["flo"], rf_signal["flo"]),
                                    (min_graph, max_graph), ls='--', c='red')
                line_r = plt.Line2D((rf_signal["fhi"], rf_signal["fhi"]),
                                    (min_graph, max_graph), ls='--', c='red')
                ax.add_line(line_l)
                ax.add_line(line_r)
            plot_time = nu.format_num(self.start_time, unit='sec')
            ax.set_title(f'({num_signals}) signals detected in PSD at t = {plot_time}')
            plt.show()
        return self.signals

    def find_bw(self, peak_i, psd_noise_level, min_prom=2, bwdb=3):
        """Find the Bandwidth for non-sll pulse.
        This method is used internally in _find_lobes method.

        Parameters
        ----------
        peak_i : float
            index of peak in frequency range.
        psd_noise_level : float
            noise level to identify raise in signal power.
        min_prom : float, optional
            minimum prominence to identify raise in signal power. The default is
            2.
        bwdb : float, optional
            Power dB to identify bandwidth point. The default is 3.

        Returns
        -------
        bw : float
            Bandwidth.
        flo : float
            low frequency.
        fhi : float
            high frequency.
        """
        # Find the Bandwidth for non-sll pulse
        # First find where signal raises above noise level at least min_prom db
        cutoff = psd_noise_level + min_prom
        cutoff_lo_i = np.where(self.psd[0:peak_i] <= cutoff)[0][-1]
        cutoff_hi_i = peak_i + np.where(self.psd[peak_i:] < cutoff)[0][0]
        local_max = signal.argrelextrema(self.psd[cutoff_lo_i:cutoff_hi_i], np.
                                         greater, order=3)
        local_max = cutoff_lo_i + local_max[0]
        if len(local_max) > 0:
            local_max_lo = local_max[0]
            local_max_hi = local_max[-1]
            if local_max_lo == local_max_hi:
                sig_mean = self.psd[local_max_lo]
            else:
                sig_mean = np.mean(self.psd[local_max_lo:local_max_hi])
            # After getting mean value of signal (taking account rippls), find
            # 3db BW
            peak_half_level = sig_mean - bwdb
            cutoff_max = max(self.psd[cutoff_lo_i], self.psd[cutoff_hi_i])
            peak_half_level = peak_half_level if peak_half_level > cutoff_max \
                else cutoff_max
            # filter out local max below peak_half_level
            local_max = local_max[np.where(self.psd[local_max] > peak_half_level)]
            flo_i = cutoff_lo_i + np.where(self.psd[cutoff_lo_i:local_max[0]] <=
                                           peak_half_level)[0][-1]
            fhi_i = local_max[-1] + np.where(self.psd[local_max[-1]:cutoff_hi_i + 1] \
                                             < peak_half_level)[0][0]
            flo = self.f_range[flo_i]
            fhi = self.f_range[fhi_i]
            bw = fhi - flo
        else: # Not a real peak found there
            bw = 0
            flo = self.f_range[peak_i]
            fhi = self.f_range[peak_i]
        return bw, flo, fhi

    def _find_lobes(self, peaks, peaks_i, peaks_f, valleys_f, prominences):
        """Check if the maximum peak is a standalone or a main lobe with side
        lobes.
        This method is used internally in get_freq_peaks method.

        Parameters
        ----------
        peaks : float array
            array of peaks in frequency domain.
        peaks_i : integer array
            indexes of peaks in frequency range.
        peaks_f : float array
            The frequency offset of each peak.
        valleys_f : float array
            The frequency offset of valleys between peaks.
        prominences : float array
            The prominences of each peak.

        Returns
        -------
        remove_peaks : Boolean array
            Array of peaks to remove from next iteration of get_freq_peaks.
        bandwidth : TYPE
            Bandwidth of maximum peak.
        flo : TYPE
            Low frequency of the peak.
        fhi : TYPE
            High frequency of the peak.
        """
        #Peak is SLL if:
        # 1. Its value is lower than main lobe
        # 2. Its value is decreasing lower than previous lobe
        # 3. The valley between lobes is keeping the same (frequency)
        # distance as between first valley and main lobe center
        # 4. There are at least 2 SLL's (on both sides)

        # Initial values
        is_main = False
        pos_main = np.argmax(peaks) # index of possible main lobe to test
        peaks_right = len(peaks_f) - pos_main - 1 # number of possible lobes to
        # the right
        peaks_left = pos_main # number of possible lobes to the left
        peaks_is_sll_r = np.array([], dtype=bool)
        peaks_is_sll_l = np.array([], dtype=bool)
        if (peaks_right > 0) and (peaks_left > 0):
            pos_main_f = peaks_f[pos_main] # frequency of possible main lobe
            #Get valleys between main lobe
            valley_r = np.where(valleys_f > pos_main_f)[0] # valleys to right
            valley_l = np.where(valleys_f < pos_main_f)[0] # valleys to left
            pos_valley_r = valley_r[0] # first valley to right
            pos_valley_l = valley_l[-1] # first valley to left

            #only compute SLL if there are valleys around main lobe
            if len(valley_r) * len(valley_l) > 0:
                tolerance = 0.35 #tolarance error of SLL width
                flo = valleys_f[pos_valley_l]
                fhi = valleys_f[pos_valley_r]
                main_dist_r = fhi - pos_main_f # Distance Main lobe to right
                # valley
                main_dist_l = pos_main_f - flo # Distance main lobe to left valley
                max_dist = max(main_dist_r, main_dist_l)
                min_dist = min(main_dist_r, main_dist_l)

                if (min_dist / max_dist) >= (1 - tolerance): # left and right sll
                    # should be same distance
                    max_sll_width = np.ceil((1 + tolerance) * max_dist)
                    min_sll_width = np.floor((1 - tolerance) * min_dist)
                else:
                    max_sll_width = 0
                    min_sll_width = 0

                if max_sll_width > 0:
                    #Frequency distances between adjacent lobes. Should be
                    #-constant for SLL
                    peaks_dist = np.diff(peaks_f)
                    peaks_dist_r = peaks_dist[pos_main + 1:]
                    right_peak_from_valley = peaks_f[pos_main + 1] - fhi
                    peaks_dist_r = np.insert(peaks_dist_r, 0,
                                             right_peak_from_valley * 2.5)
                    peaks_dist_l = peaks_dist[0:pos_main][::-1] if pos_main > 0 else \
                        np.array([])
                    left_peak_from_valley = flo - peaks_f[pos_main - 1]
                    peaks_dist_l = np.append(peaks_dist_l, left_peak_from_valley *
                                             2.5)

                    # Check if the peaks are within max_sll_width distance
                    peaks_dist_is_sll_r = np.logical_and(peaks_dist_r <=
                                                         max_sll_width,
                                                         peaks_dist_r >=
                                                         min_sll_width)
                    peaks_dist_is_sll_l = np.logical_and(peaks_dist_l <=
                                                         max_sll_width,
                                                         peaks_dist_l >=
                                                         min_sll_width)

                    #Check if peaks are in descending order from main lobe (positive peaksLevel)
                    #allow a minimum glitch of 5% of prominence
                    peaks_level_dist = np.diff(peaks)
                    glitch = 0.10 * prominences
                    peaks_level_r = peaks_level_dist[pos_main:] + glitch[
                        pos_main + 1:]
                    peaks_level_l = peaks_level_dist[0:pos_main][::-1] + glitch[0:
                        pos_main][::-1]

                    peaks_level_is_sll_r = peaks_level_r <= 0
                    peaks_level_is_sll_l = peaks_level_l >= 0

                    # Check if peaks answer both conditions
                    peaks_is_sll_r = np.logical_and(peaks_level_is_sll_r,
                                                    peaks_dist_is_sll_r)
                    peaks_is_sll_l = np.logical_and(peaks_level_is_sll_l,
                                                    peaks_dist_is_sll_l)

                    #Check conditions for main lobe minimum 2 SLL
                    is_main = peaks_is_sll_r[0] and peaks_is_sll_l[0]

        # Define bandwidth
        if is_main: # Phased signal with main lobe and side lobes
            bandwidth = main_dist_l + main_dist_r
            remove_peaks = np.array([*peaks_is_sll_l, True, *peaks_is_sll_r])
            # Flo, Fhi should be defined from main lobe and first SLL
            fhi = peaks_f[pos_main] + main_dist_r
            flo = peaks_f[pos_main] - main_dist_l
        else: # Signal with no side lobes.
            noise_level = peaks[pos_main] - prominences[pos_main]
            bandwidth, flo, fhi = self.find_bw(peaks_i[pos_main],
                                               noise_level, min_prom=prominences
                                               [pos_main] / 2)
            remove_peaks = np.logical_and(peaks_f > flo, peaks_f < fhi)

        return remove_peaks, bandwidth, flo, fhi

    def moving_average(self, N, vec='abs', showplot=False):
        """Calculate Moving Average of N samples for specified vector.

        Parameters
        ----------
        N : integer
            Window size of moving average.
        vec : string, optional
            Indicate what data vector to calculate MA. The default is 'abs'.
            possible values:
                'abs' : absolute value of IQ (default)
                'real' : real value of IQ
                'imag' : imag value of IQ
                'iq' : complex value of IQ
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        ma : float array
            Moving Average result.
        """
        if isinstance(vec, str):
            if vec == 'abs':
                x = self.get_abs()
            elif vec == 'real':
                x = self.iq.real
            elif vec == 'imag':
                x = self.iq.imag
            elif vec == 'iq':
                x = self.iq
            else:
                x = self.get_abs()
        elif isinstance(vec, np.ndarray):
            x = vec
        elif isinstance(vec, list):
            x = np.array(vec)
        else:
            x = self.get_abs()

        tail = x[-N + 1:]
        tail_avg = np.average(tail)
        x = np.append(x, np.ones(N - 1) * tail_avg)
        cumsum = np.cumsum(np.insert(x, 0, 0))
        ma = (cumsum[N:] - cumsum[:-N]) / float(N)
        if showplot:
            plt.figure()
            plt.plot(x, 'b', label='Original')
            plt.plot(ma, 'g', label='Moving average')
            delay = nu.format_num(self.ts * N, 2, 'sec')
            plt.title(f'Moving average. N={N} ({delay})')
            plt.grid()
            plt.legend()
            plt.show()
        return ma

    def get_pulses(self, top=0.4, showplot=False):
        """Find pulses in time domain.
        Detected pulses are minimum 40% (default) below maximum signal level.

        Parameters
        ----------
        top : float, optional
            Percentage below maximum signal level to search pulses. Default is
            0.4.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        widths : float array
            Array of pulses widths.
        left_ips : float array
            Times of left side (rise) of pulses.
        right_ips : float array
            Times of right side (drop) of pulses.
        pulse_height : float array
            Array of pulses height.
            Each height is the mean value between rise (left) and drop (right)
        prt : float
            Pulse Repetition Time.
        """
        # Find pulses in timeline
        #Use Moving Average to filter signal
        #Pulses are minimum 40% (default -np.array(top)) below max signal level
        min_w = 300 # minimum pulse width in samples
        min_dist = 3 * min_w # minimum distance between pulses in samples
        dat = self.get_abs()
        dat_ma = self.moving_average(min_w, 'abs')
        dat_ma_min = min(dat_ma)
        dat_ma_max = max(dat_ma)
        dat_amp = dat_ma_max - dat_ma_min
        dat_top = dat_ma_max - top * dat_amp
        dat_bin = np.multiply((dat_ma > dat_top), np.ones(len(dat_ma)) * dat_amp)

        pks_dict = signal.find_peaks(dat_bin, width=min_w,
                                     distance=min_dist)[1]
        # Search for ips's: pulses left and right edges
        left_ips_s = np.round(pks_dict['left_ips']).astype(int)
        left_ips_e = np.round(left_ips_s + min_w).astype(int)
        right_ips_s = np.round(pks_dict['right_ips']).astype(int)
        right_ips_e = np.round(right_ips_s + min_w).astype(int)

        if len(left_ips_s) > 0:
            if right_ips_e[-1] > len(dat):
                right_ips_e[-1] = len(dat)
            left_ips = []
            right_ips = []
            for i in range(len(left_ips_s)):
                l_ips = np.where(dat[left_ips_s[i]:left_ips_e[i]] > dat_top)[0][0] \
                    + left_ips_s[i]
                left_ips.append(l_ips)
                r_ips = np.where(dat[right_ips_s[i]:right_ips_e[i]] > dat_top)[0] \
                    [-1] + right_ips_s[i]
                right_ips.append(r_ips)

            left_ips = np.array(left_ips)
            right_ips = np.array(right_ips)
            widths = (right_ips - left_ips) * self.ts
            #widths = pks_dict['widths'] * self.ts

            pulse_height = []
            for i in range(len(left_ips)):
                height = np.sqrt(np.mean(dat[left_ips[i]:right_ips[i]]**2)) # RMS
                pulse_height.append(height)
            left_ips = self.ts * left_ips
            right_ips = self.ts * right_ips
        else:
            widths = []
            left_ips = []
            right_ips = []
            pulse_height = []

        #Identify recurring pulses and calculate real RPT
        #overcome problem with reflected pulses
        num_pulses = len(widths)
        pw = np.median(widths) if num_pulses > 0 else 0

        if num_pulses < 2:
            prt = 0
        elif num_pulses == 2:
            prt = left_ips[1] - left_ips[0]
            prt = prt if (prt > 2 * pw) else 0 #pulses couldn't be too close
        elif num_pulses == 3:
            dist_1_0 = left_ips[1:] - left_ips[0:-1] # distance between adjacent
            # pulses
            stdev1 = np.std(dist_1_0)
            eps = pw / 2 # epsilon
            prt = np.mean(dist_1_0) if stdev1 < eps else 0
        elif num_pulses > 3:
            dist_1_0 = left_ips[1:] - left_ips[0:-1] # distance between adjacent
            # pulses
            dist_2_0 = left_ips[2:] - left_ips[0:-2] # distance between every
            # second pulse
            stdev1 = np.std(dist_1_0)
            stdev2 = np.std(dist_2_0)
            eps = pw / 2 # epsilon
            if stdev1 < eps:
                prt = np.median(dist_1_0)
            elif stdev2 < eps:
                prt = np.median(dist_2_0)
            elif num_pulses >= 10: # if enough pulses
                # get PRT from majority
                low = np.percentile(dist_1_0, 20)
                hi = np.percentile(dist_1_0, 90)
                dist_1_0 = dist_1_0[dist_1_0 > low]
                dist_1_0 = dist_1_0[dist_1_0 < hi]
                stdev1 = np.std(dist_1_0)
                prt = np.median(dist_1_0) if stdev1 < eps else 0
            else:
                prt = 0
        else:
            prt = 0

        prt = round(prt, 6)

        if showplot:
            plt.figure()
            plt.axes()
            plt.plot(self.t, dat, 'b')
            plt.plot(self.t, dat_ma, 'g-')
            if len(widths) > 0:
                for i in range(len(left_ips)):
                    line_l = plt.Line2D((left_ips[i], left_ips[i]),
                                        (dat_ma_min, pulse_height[i]), ls='--', c=
                                            'red')
                    line_r = plt.Line2D((right_ips[i], right_ips[i]),
                                        (dat_ma_min, pulse_height[i]), ls='--', c=
                                            'red')
                    line_top_avg = plt.Line2D((left_ips[i], right_ips[i]),
                                              (pulse_height[i], pulse_height[i]),
                                              ls='--', c='red')
                    plt.gca().add_line(line_l)
                    plt.gca().add_line(line_r)
                    plt.gca().add_line(line_top_avg)

                line_top = plt.Line2D((self.t[0], self.t[-1]),
                                      (dat_top, dat_top), ls='--', c='black')
                plt.gca().add_line(line_top)
                plt.title(f'({len(widths)}) Pulses. PRT ({prt})')
            else:
                plt.title('No Pulses Detected')
            plt.show()
        return widths, left_ips, right_ips, pulse_height, prt

    def get_noise_level(self, w_len=400, showplot=False):
        """Calculate white gaussian noise level.
        Measure std-dev in rolling window size w-len through all the signal.
        The noise in the minimum std-dev along the IQ.

        Parameters
        ----------
        w_len : integer, optional
            Window size in samples. The default is 400.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).
        Returns
        -------
        noise : float
            Signal noise level.
        stdev : float array
            Full array of std-dev.
        """
        # calculate noise where there are no pulses in IQ
        # Measure std-dev in rolling window size w-len through all the signal
        stdev = pd.Series(self.iq.real).rolling(w_len).std()
        noise = np.sqrt(2) * stdev.min()

        if showplot:
            dat_ma = self.moving_average(w_len, 'abs')
            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(self.t, self.iq.real, 'b', label='Signal')
            ax.plot(self.t, np.nan_to_num(stdev), 'g', label='Std Dev')
            ax.plot(self.t, dat_ma, 'c', label='MA')
            line_noise = plt.Line2D((self.t[0], self.t[-1]), (noise, noise),
                                    ls='--', c='red', label='Noise')
            plt.gca().add_line(line_noise)
            ax.set_title('Signal and Noise Level')
            plt.legend()
            plt.show()
        return noise, stdev

    def get_signal_power(self, percentile=90, showplot=False):
        """Calculate power and RMS of signal.

        Parameters
        ----------
        percentile : integer, optional
            Top signal cutoff percentile. The default is 90.
        showplot : Boolean, optional
            Show a plot of the result. The default is False (no plot).

        Returns
        -------
        signal_power : float
            Power of signal (calculate only top percentile of signal).
        signal_rms : float
            Signal RMS sqrt of power.
        """
        cutoff = np.percentile(self.get_abs(), percentile)
        signal_top = self.abs[np.where(self.abs > cutoff)]
        signal_power = np.mean(signal_top**2)
        signal_rms = np.sqrt(signal_power)

        if showplot:
            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(self.t, self.iq.real, 'g', label='Signal')
            line_cutoff = plt.Line2D((self.t[0], self.t[-1]), (cutoff, cutoff),
                                     ls='--', c='red', label=f'Cutoff {nu.
                                         format_num(percentile)}')
            line_rms = plt.Line2D((self.t[0], self.t[-1]), (signal_rms,
                                                             signal_rms),
                                  ls='--', c='black', label=f'rms={nu.
                                      format_num(signal_rms)}')
            line_power = plt.Line2D((self.t[0], self.t[-1]), (signal_power,
                                                               signal_power),
                                    ls='--', c='blue', label=f'power={nu.
                                        format_num(signal_power)} W')
            plt.gca().add_line(line_cutoff)
            plt.gca().add_line(line_rms)
            plt.gca().add_line(line_power)
            ax.set_title(f'Signal Power and RMS')
            plt.legend()
            plt.show()
        return signal_power, signal_rms

    def generate_awgn(self, noise, method='relative'):
        """Generate additive white gaussian noise and add it to self.iq.

        Parameters
        ----------
        noise : integer, float or string.
            Depending on method, indicate the amount of noise to generate.
            If it's a string in the form '4db' then it's considered as
            logarithmic dB.
            Otherwise the input will be considered as a linear number.
        method : string, optional (Default is 'relative')
            The method of generated noise calculation. The default is 'relative'.
            'relative' : Multiply of current noise level. Current noise will be
                         calculated and multiplied.
            'absolute' : The input value is the required generated noise. Add it
                         to current noise.
            'snr' : Calculate maximum current SNR, and add more noise to reach
                    new snr (should be lower)

        Returns
        -------
        new_noise : float
            New noise level that was added to the current noise.
        current_noise : float
            Current noise level, before adding new noise.
        """
        if isinstance(noise, str):
            noise = nu.format_num(noise) # convert db to linear
        current_noise, _ = self.get_noise_level()
        new_noise = 0

        if method == 'relative':
            if noise > 1: # noise is multiply factor
                new_noise = noise * current_noise
        elif method == 'absolute':
            if noise > current_noise: # noise is required noise level
                new_noise = noise
        elif method == 'snr':
            sig_power, sig_rms = self.get_signal_power()
            current_snr = sig_power / (current_noise ** 2)
            if current_snr > noise: # noise is actually required snr in this case
                new_noise = np.sqrt(sig_power / noise)

        if new_noise > current_noise:
            add_noise = np.sqrt(new_noise ** 2 - current_noise ** 2) / np.sqrt(2)
            noise_real = np.random.default_rng().normal(self.iq.real, add_noise)
            noise_imag = np.random.default_rng().normal(self.iq.imag, add_noise)
            self.iq = noise_real + 1j * noise_imag
        return new_noise, current_noise

    def plot_iq(self, **kwargs):
        """Generate plots of IQ data. Can either show on screen or save PNG file.
        Can generate several graphs on the same plot.

        Parameters
        ----------
        **kwargs : dict
            save=True. Save the plot to PNG file (default is False)
            file='file_name'. Name of the PNG file to save. Default is plot
            theme time.
            path='file_path'. Path destination of the PNG file. Default is
            current dir.
            plot1='time.real'. Plot real part of IQ vector (time domain)
            plot2='time.imag'. Plot imaginary part of IQ vector (time domain)
            plot3='time.abs'. Plot absolute of IQ vector (time domain)
            plot4='psd'. Plot Welch Power Spectral Density.

        Returns
        -------
        None.

        """
        #Available plots: time.real, time.imag, time.abs, psd (or freq)
        num_plots = sum(['plot' in i for i in kwargs.keys()])
        if num_plots == 0:
            kwargs.update({'plot1': 'time.real'})
            num_plots = 1

        plot_time = nu.format_num(self.start_time, unit='sec')
        plot_time_str = f' at t = {plot_time}'
        filename = ""
        filepath = ""
        xlabel = ""
        ylabel = ""
        name = ""
        title = ""
        saveplot = False

        fig, ax = plt.subplots(num_plots, 1)
        plotnum = 1
        for key, val in kwargs.items():
            if key == 'file':
                filename = val
            elif key == 'path':
                filepath = val
            elif key == 'save':
                saveplot = val
            elif 'plot' in key:
                if val.find('time') >= 0:
                    ax = plt.subplot(num_plots, 1, plotnum)
                    plotnum = plotnum + 1
                    xlabel = 'Time [sec]'
                    ylabel = 'Amplitude'
                    name = 'time'
                    if val.find('real') >= 0:
                        d = self.iq.real
                        title = 'Time Domain (real part)'
                        name = 'time_real'
                    elif val.find('imag') >= 0:
                        d = self.iq.imag
                        title = 'Time Domain (imag part)'
                        name = 'time_imag'
                    elif val.find('abs') >= 0:
                        d = self.get_abs()
                        title = 'Time Domain (abs value)'
                        name = 'time_abs'
                    else:
                        d = self.iq.real
                        title = 'Time Domain (real part)'
                        name = 'time_real'
                    ax.plot(self.t, d)

                elif val == 'psd' or val == 'freq':
                    if not hasattr(self, 'psd'):
                        self.pwelch()
                    ax = plt.subplot(num_plots, 1, plotnum)
                    plotnum = plotnum + 1
                    xlabel = 'Freq [Hz]'
                    ylabel = 'PSD [dB/Hz]'
                    title = 'Power Spectrum Density'
                    ax.plot(self.f_range, self.psd, 'b')
                    name = 'PSD_'

                    if hasattr(self, 'signals'):
                        min_graph = min(self.psd)
                        max_graph = max(self.psd)
                        for i in self.signals:
                            rf_signal = self.signals[i]
                            line_l = plt.Line2D(
                                (rf_signal["flo"], rf_signal["flo"]),
                                (min_graph, max_graph), ls='--', c='red'
                            )
                            line_r = plt.Line2D(
                                (rf_signal["fhi"], rf_signal["fhi"]),
                                (min_graph, max_graph), ls='--', c='red'
                            )
                            ax.add_line(line_l)
                            ax.add_line(line_r)
                        if len(self.signals) > 0:
                            title = title + f'. ({len(self.signals)}) signals detected in PSD'

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title + plot_time_str)
                ax.grid()
        plt.tight_layout(h_pad=1.0)
        if saveplot:
            if filename == "":
                filename = name + plot_time + '.png'
            elif filename[-4:] != '.png':
                filename = filename + '.png'
            if os.path.isdir(filepath):
                fullpath = os.path.join(filepath, filename)
            else:
                fullpath = os.path.join(os.path.dirname(filepath), filename)
            plt.savefig(fullpath)
            plt.close()
        else:
            plt.show()


#################
def main():
    start_time = 0
    sig_length = 1e-3

    xdat_file = 'C:\IQ_Data\Radar\Kisufim\21_RADA+KIPA2020-01-01_23-35-25.xdat'
    myIQ = IQData(xdat_file, start_time, sig_length)

    myIQ.pwelch()

    distance_percent = 0.005
    min_distance = round(distance_percent * len(myIQ.f_range))
    min_width = np.floor(min_distance / 2)
    min_prominence = 2
    peaks2, peaks_i2, peaks_f2, valleys_f2, prominences2 = \
        myIQ.get_freq_peaks(min_distance,
                            min_width,
                            min_prominence, showplot=True)

    distance_percent = 0.01
    min_distance = round(distance_percent * len(myIQ.f_range))
    min_width = np.floor(min_distance / 2)
    min_prominence = 3
    peaks3, peaks_i3, peaks_f3, valleys_f3, prominences3 = \
        myIQ.get_freq_peaks(min_distance,
                            min_width,
                            min_prominence, showplot=True)

    distance_percent = 0.05
    min_distance = round(distance_percent * len(myIQ.f_range))
    min_width = np.floor(min_distance / 2)
    min_prominence = 7
    peaks7, peaks_i7, peaks_f7, valleys_f7, prominences7 = \
        myIQ.get_freq_peaks(min_distance,
                            min_width,
                            min_prominence, showplot=True)
    pass

if __name__ == '__main__':
    main()