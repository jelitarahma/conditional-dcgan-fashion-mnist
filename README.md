# Conditional DCGAN Fashion-MNIST

Implementasi Conditional Deep Convolutional Generative Adversarial Network (cDCGAN) pada dataset Fashion-MNIST serta evaluasi performa klasifikasi menggunakan Convolutional Neural Network (CNN).

---

## Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/jelitarahma/conditional-dcgan-fashion-mnist.git
cd conditional-dcgan-fashion-mnist
```

### 2. Buat dan Aktifkan Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (CMD):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Cara Menjalankan

### 1. Pipeline Utama (Full Run)

#### A. Melatih Conditional DCGAN (20 Epoch)
Melatih model Generator dan Discriminator pada dataset Fashion-MNIST:
```bash
python -m src.train_gan --config configs/dcgan.yaml --run-name dcgan_full_run
```
Output tersimpan di `outputs/dcgan_full_run/` (termasuk kurva loss dan checkpoint model).

#### B. Menghasilkan Citra Sintetis (Generate Dataset)
Membuat 30.000 citra sintetis berlabel seimbang (3.000 per kelas) menggunakan bobot generator yang sudah dilatih:
```bash
python -m src.generate --checkpoint outputs/dcgan_full_run/checkpoints/generator_final.pt --run-name dcgan_full_run --samples-per-class 3000
```
Hasil grid preview tersimpan di `outputs/dcgan_full_run/synthetic/preview_grid.png`.

#### C. Melatih & Mengevaluasi Model CNN (4 Skenario × 3 Seed)
Melatih classifier CNN untuk membandingkan performa data riil dan data sintetis (V0: 30k riil, V1: 60k riil, V2: 30k riil + 30k sintetis, V3: 60k riil + 30k sintetis):
```bash
python -m src.train_cnn --config configs/cnn.yaml --run-name dcgan_full_run --variant all --seeds 0 1 2
```

#### D. Evaluasi Standalone & Pengukuran Kualitas Sintetis
Mengevaluasi akurasi test set dan matriks konfusi:
```bash
python -m src.evaluate --config configs/cnn.yaml --run-name dcgan_full_run --variants all --seeds 0 1 2
```
Mengukur seberapa baik citra sintetis dapat dikenali oleh model CNN yang dilatih pada citra riil:
```bash
python -m src.evaluate --config configs/cnn.yaml --run-name dcgan_full_run --eval-synthetic-with-real
```

---

### 2. Eksplorasi Hyperparameter (Ablation Study)

#### A. Menjalankan Seluruh Sweep Otomatis
Menjalankan 9 variasi eksperimen hyperparameter (dimensi laten $nz$, filter $ngf/ndf$, arsitektur layer, slope aktivasi LeakyReLU, learning rate, dan batch size):
```bash
python scripts/run_hparam_sweep.py
```
Ringkasan perbandingan hasil otomatis tersimpan di `outputs/hparam_sweep_summary.csv`.

#### B. Menjalankan Variasi Hyperparameter Tertentu (Manual via CLI)
Anda dapat memodifikasi parameter konfigurasi menggunakan flag `--override`:

* **Eksplorasi Dimensi Laten ($nz = 50$ atau $nz = 200$):**
  ```bash
  python -m src.train_gan --config configs/dcgan.yaml --run-name hparam_nz_50 --override model.nz=50
  python -m src.train_gan --config configs/dcgan.yaml --run-name hparam_nz_200 --override model.nz=200
  ```

* **Eksplorasi Kapasitas Filter ($ngf = ndf = 32$):**
  ```bash
  python -m src.train_gan --config configs/dcgan.yaml --run-name hparam_filters_32 --override model.ngf=32 --override model.ndf=32
  ```

* **Eksplorasi Kedalaman Konvolusi Tambahan & Ukuran Kernel:**
  ```bash
  python -m src.train_gan --config configs/dcgan.yaml --run-name hparam_extra_layers_1 --override model.extra_layers=1
  python -m src.train_gan --config configs/dcgan.yaml --run-name hparam_kernel_5 --override model.extra_layers=1 --override model.kernel_size=5
  ```

* **Eksplorasi Slope LeakyReLU ($slope = 0.1$):**
  ```bash
  python -m src.train_gan --config configs/dcgan.yaml --run-name hparam_slope_01 --override model.leaky_relu_slope=0.1
  ```

* **Eksplorasi Learning Rate ($lr = 0.0005$):**
  ```bash
  python -m src.train_gan --config configs/dcgan.yaml --run-name hparam_lr_5e4 --override training.lr_g=0.0005 --override training.lr_d=0.0005
  ```

* **Eksplorasi Ukuran Batch ($batch\_size = 64$):**
  ```bash
  python -m src.train_gan --config configs/dcgan.yaml --run-name hparam_batch_64 --override data.batch_size=64
  ```


---

## Output Proyek
Hasil eksekusi disimpan di folder `outputs/`:
* `outputs/dcgan_full_run/plots/loss_curve.png`: Grafik kurva loss Generator dan Discriminator (20 epoch).
* `outputs/dcgan_full_run/samples/`: Gambar sampel progresif per checkpoint epoch (`epoch_005.png` s.d. `epoch_020.png`).
* `outputs/dcgan_full_run/synthetic/preview_grid.png`: Grid sampel citra sintetis untuk seluruh 10 kelas.
* `outputs/dcgan_full_run/metrics/results_summary.csv`: Ringkasan akurasi, Macro-F1, Precision, dan Recall pengujian CNN 4 skenario.
* `outputs/hparam_sweep_summary.csv`: Tabel perbandingan akurasi sintetis seluruh variasi eksplorasi hyperparameter.
