import numpy as np
import librosa
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
AUDIO_PATH = "le1f_wut.wav"
HOP_LENGTH = 512
GRID_EVERY_SECONDS = 20  # spacing for approximate time markers
BASS_COLOR = "#3A3A3A"
SEED = 13
np.random.seed(SEED)

# ----------------------------
# LOAD AUDIO
# ----------------------------
y, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)
duration = librosa.get_duration(y=y, sr=sr)

# ----------------------------
# STFT
# ----------------------------
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP_LENGTH))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP_LENGTH)


# ----------------------------
# FEATURE HELPERS
# ----------------------------
def band_energy(S, freqs, fmin, fmax):
    """Compute average energy in a frequency band over time."""
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(idx) == 0:
        return np.zeros(S.shape[1])
    return S[idx, :].mean(axis=0)


def norm(x):
    """Normalize array to [0, 1]."""
    x = np.asarray(x)
    if np.allclose(x.max(), x.min()):
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


# ----------------------------
# FEATURES
# ----------------------------
# Low-frequency energy (bass range)
bass = band_energy(S, freqs, 25, 110)

# Upper-mid frequency band (rough vocal range)
mid = band_energy(S, freqs, 500, 2500)

# Onset strength used to capture rhythmic activity
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

# Spectral flatness as a proxy for texture / noise
flatness = librosa.feature.spectral_flatness(S=S)[0]

bass_n = norm(bass)
mid_n = norm(mid)
onset_n = norm(onset_env)
flat_n = norm(flatness)

# Identify low-energy regions (used as drops)
overall = norm(bass + mid + onset_env)
drop_mask = overall < np.quantile(overall, 0.20)

# ----------------------------
# VISUAL SETUP
# ----------------------------
fig = plt.figure(figsize=(16, 8), dpi=200)
ax = plt.gca()
ax.set_facecolor("white")

# Subtle vertical time guides
for gt in np.arange(0, duration + 1e-9, GRID_EVERY_SECONDS):
    ax.axvline(gt, linewidth=0.8, alpha=0.08)

# Time labels along the bottom
for gt in np.arange(0, duration + 1e-9, GRID_EVERY_SECONDS):
    mm = int(gt // 60)
    ss = int(gt % 60)
    ax.text(
        gt,
        -0.10,
        f"{mm}:{ss:02d}",
        fontsize=8,
        alpha=0.55,
        ha="center",
        transform=ax.get_xaxis_transform(),
    )

# Vertical layout positions for each layer
Y_TEXTURE = 0.78
Y_RHYTHM = 0.60
Y_MID = 0.40
Y_BASS = 0.16

# ----------------------------
# TEXTURE LAYER
# ----------------------------
# Draw translucent texture regions based on spectral flatness
chunk = int((sr / HOP_LENGTH) * 7)
for start in range(0, len(times), chunk):
    end = min(start + chunk, len(times) - 1)
    lvl = flat_n[start:end].mean()
    if lvl < 0.30:
        continue

    t0, t1 = times[start], times[end]
    xs = np.linspace(t0, t1, 10)
    jitter = (np.random.rand(len(xs)) - 0.5) * 0.04
    top = Y_TEXTURE + 0.07 * lvl + jitter
    bot = Y_TEXTURE - 0.07 * lvl + jitter

    ax.fill_between(xs, bot, top, alpha=0.12, linewidth=0)

# ----------------------------
# Rhythm marks (intentionally imprecise)
# ----------------------------
idx = np.where(onset_n > 0.28)[0]

for i in idx:
    if drop_mask[i]:
        continue
    if np.random.rand() < 0.35:  # randomly skip some hits
        continue

    t = times[i] + (np.random.rand() - 0.5) * 0.08
    h = (0.04 + 0.14 * onset_n[i]) * (0.7 + np.random.rand() * 0.6)

    ax.plot([t, t], [Y_RHYTHM - h / 2, Y_RHYTHM + h / 2], linewidth=1.1, alpha=0.75)


# ----------------------------
# MIDBAND MOTION (TEXTURAL, NOT MEASUREMENT)
# ----------------------------
# Smooth heavily so it reads as motion, not waveform
window = 35
mid_smooth = np.convolve(mid_n, np.ones(window) / window, mode="same")

mid_line = Y_MID + (mid_smooth - 0.5) * 0.18
mid_line += np.random.randn(len(mid_line)) * 0.008

# Hard gaps during drops so silence is obvious
mid_line[drop_mask] = np.nan

# Main trace (very light)
ax.plot(times, mid_line, linewidth=2.0, alpha=0.35)

# Very faint echoes
for k, a in [(3, 0.12), (6, 0.07)]:
    echo = np.roll(mid_line, k)
    ax.plot(times, echo, linewidth=1.5, alpha=a)

# ----------------------------
# BASS BLOCKS (WEIGHT / MASS)
# ----------------------------
win = int((sr / HOP_LENGTH) * 3.0)

for start in range(0, len(times), win):
    end = min(start + win, len(times) - 1)
    lvl = bass_n[start:end].mean()

    if lvl < 0.28:
        continue
    if drop_mask[start:end].mean() > 0.5:
        continue

    t0, t1 = times[start], times[end]
    xs = np.linspace(t0, t1, 8)

    jag = (np.random.rand(len(xs)) - 0.5) * 0.10
    top = Y_BASS + 0.05 + 0.35 * lvl + jag
    bot = np.full_like(xs, Y_BASS - 0.03)

    ax.fill_between(xs, bot, top, color=BASS_COLOR, alpha=0.85, linewidth=0)


# ----------------------------
# EMPHASIZE SILENCE / DROPS
# ----------------------------
drop_times = times[drop_mask]

for t in drop_times[::30]:
    ax.axvspan(t - 0.2, t + 0.2, color="white", alpha=0.9, zorder=10)

# ----------------------------
# FINAL TOUCHES
# ----------------------------
ax.set_xlim(0, duration)
ax.set_ylim(0, 1)
ax.set_yticks([])

ax.set_title(
    "Score for Listening: Le1f – Wut\n"
    "A listening map emphasizing bass weight, rhythmic pressure, texture, and drops",
    fontsize=14,
)

for spine in ax.spines.values():
    spine.set_alpha(0.12)

# Subtle annotation to highlight a structural moment
ax.text(
    95,  # adjust slightly if needed
    0.52,  # vertical position between layers
    "drop",
    fontsize=10,
    alpha=0.5,
    ha="center",
)

# Header metadata (course + assignment info)
fig.text(
    0.01,
    0.97,
    "MUSC1001 · Professor Gutierrez · 02-01-2026\n"
    "Connor Tessaro · Describing Music Visually Assignment",
    fontsize=9,
    alpha=0.7,
    ha="left",
    va="top",
)


plt.tight_layout()
plt.savefig("score-for-listening_le1f-wut.png", bbox_inches="tight")
plt.show()

print("Saved: score-for-listening_le1f-wut.png")
