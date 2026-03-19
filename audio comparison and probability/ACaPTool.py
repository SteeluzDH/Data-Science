import customtkinter as ctk
from tkinter import filedialog, Toplevel
import numpy as np
import librosa
from librosa.sequence import dtw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf


ctk.set_appearance_mode("dark")

class AudioComparer:
    def __init__(self, root):
        self.root = root
        self.root.title("ACaP Tool")
        
        self.audio1 = None
        self.audio2 = None
        self.sr1 = None
        self.sr2 = None
        
        #GUI
        self.frame = ctk.CTkFrame(root)
        self.frame.pack(fill="both", expand=True)
        
        self.load_btn1 = ctk.CTkButton(self.frame, text="Load Audio File 1", command=self.load_audio1)
        self.load_btn1.pack(pady=10)
        
        self.load_btn2 = ctk.CTkButton(self.frame, text="Load Audio File 2", command=self.load_audio2)
        self.load_btn2.pack(pady=10)
        
        self.compare_btn = ctk.CTkButton(self.frame, text="Compare Audio Files", command=self.compare_audio)
        self.compare_btn.pack(pady=10)
        
        #Matplot lib figure
        self.fig, self.ax = plt.subplots(2, 1, figsize=(8,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def load_audio1(self):
        path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if path:
            self.audio1, self.sr1 = librosa.load(path, sr=None)
            self.plot_waveforms()
    
    def load_audio2(self):
        path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if path:
            self.audio2, self.sr2 = librosa.load(path, sr=None)
            self.plot_waveforms()
    
    def plot_waveforms(self):
        self.ax[0].clear()
        self.ax[1].clear()
        
        if self.audio1 is not None:
            self.ax[0].plot(self.audio1, color='blue')
            self.ax[0].set_title("Audio File 1")
        
        if self.audio2 is not None:
            self.ax[1].plot(self.audio2, color='green')
            self.ax[1].set_title("Audio File 2")
        
        self.canvas.draw()
        
        
    #===ML Part===
    def extract_features(self, audio, sr):
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = mfcc.T   # (time, features)
        return mfcc
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32)
        ])
        return model
    
    
    def compare_audio(self):
        if self.audio1 is None or self.audio2 is None:
            return

        mfcc1 = self.extract_features(self.audio1, self.sr1)
        mfcc2 = self.extract_features(self.audio2, self.sr2)

        #DTW alignment
        D, wp = dtw(mfcc1.T, mfcc2.T, metric='cosine')

        wp = np.array(wp)

        matches = []

        #Normalising similarity
        max_dist = np.max(D)

        for i, j in wp:
            dist = D[i, j]
            similarity = 1 - (dist / max_dist)

            if similarity > 0.7:  #lowered threshold
                matches.append((i, j, similarity))

        self.show_results(matches)
    
    def show_results(self, matches):
        win = Toplevel(self.root)
        win.title("Comparison Results")
        
        fig, ax, = plt.subplots(figsize=(8,4))
        
        ax.plot(self.audio1, alpha=0.5, label="Audio 1")
        ax.plot(self.audio2, alpha=0.5, label="Audio 2")
        
        #drawing rings to show matches
        hop_length = 512

        for (i, j, sim) in matches[::5]:  #skip every 5th for performance
            x = int(i * hop_length)
            
            if x < len(self.audio1):
                y = self.audio1[x]

                ax.scatter(
                    x,
                    y,
                    s=200,
                    facecolors='none',
                    edgecolors='red',
                    alpha=0.6
                )
        
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

if __name__ == "__main__":
    root = ctk.CTk()
    app = AudioComparer(root)
    root.mainloop()