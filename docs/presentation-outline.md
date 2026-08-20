# Outline Presentasi: Ekstraksi Field KYC dari Foto Dokumen Identitas

Semua angka dalam format **val / test**, satuan persen.

Kecuali Slide 1 bagian preprocessing (diberi label eksplisit), **seluruh angka diukur pada split
yang sama** (train 535 / val 49 / test 48), jadi bisa dibandingkan langsung antar slide.

Metrik yang dipakai:

- **CER** (Character Error Rate), proporsi karakter yang perlu diubah agar jawaban sama persis. Makin kecil makin baik. Bisa lewat 100% kalau tebakan jauh lebih panjang dari jawaban benar.
- **WER** (Word Error Rate), sama tapi dihitung per kata.
- **Exact match**, persentase jawaban yang sama persis. Metrik paling keras.

---

# BAGIAN 1: MODELING

## Slide 1: Zero-shot baseline dan usaha pertama memperbaikinya

**Pesan utama:** Satu field selesai lewat insight EDA. Dua sisanya buruk, dan memperbaiki gambar
ternyata tidak menolong.

### 1a. Baseline zero-shot

Pipeline: foto mentah → PaddleOCR → regex dan heuristik. Nol training, nol anotasi manual.

| Field | CER | WER | Exact match |
|---|---|---|---|
| birth_date | 6.9 / 8.3 | 7.5 / 8.3 | **89.8 / 91.7** ✅ |
| name | 50.2 / 54.7 | 63.4 / 73.4 | 20.4 / 16.7 ❌ |
| address | 77.4 / 79.9 | 84.7 / 85.8 | **0.0 / 0.0** ❌ |

birth_date langsung beres karena EDA menemukan tanggal lahir tertanam di 6 digit pertama nomor
MyKad. Satu aturan regex, selesai.

> Catatan kaki opsional: ablasi terkontrol pada split yang sama menunjukkan aturan itu menyumbang
> 92 poin CER. Tanpanya, tidak ada regex tanggal generik yang menemukan apapun (exact match 0%).

### 1b. Usaha pertama: perbaiki gambarnya

⚠️ *Diukur pada generasi split yang lebih awal. Perbandingan valid antar baris di tabel ini saja,
jangan disandingkan dengan tabel slide lain.*

| Preprocessing | CER nama | WER nama | Exact nama |
|---|---|---|---|
| Tanpa preprocessing (pembanding) | 55.2 / 59.6 | 69.3 / 74.9 | **19.0 / 15.9** |
| Grayscale | 55.1 / 58.3 | 71.1 / 73.8 | 11.1 / 12.7 |
| Koreksi perspektif | 53.0 / 53.7 | 70.1 / 67.6 | 12.7 / 15.9 |
| Koreksi perspektif (Hough) | 58.6 / 55.4 | 72.3 / 64.4 | 12.7 / 17.5 |
| Koreksi perspektif (LDRNet) | 82.5 / 86.0 | 91.7 / 92.1 | 0.0 / 3.2 |
| Pipeline preprocessing penuh | 56.9 / - | 80.6 / - | 7.9 / - |

**Talking point:** CER bergerak sedikit ke sana ke mari, tapi **exact match turun di semua varian**
pada val. Tidak ada satupun yang benar benar membantu, dan versi otomatis (LDRNet) malah rusak parah
karena sering salah memotong kartu.

Kesimpulannya: kalau memperbaiki gambar tidak menggerakkan angka, berarti asumsi kami soal letak
masalahnya yang salah.

---

## Slide 2: Is it the OCR, or the Parser?

**Pesan utama:** Kami ukur langsung, dan jawabannya berbeda untuk tiap field.

### Bukti kualitatif (pimpin dengan ini)

Contoh nyata, semuanya dari eval set:

| Gambar | Ground truth | Baseline0 memprediksi |
|---|---|---|
| image_064 | AFIZAN BIN ZAINAL ABIDIN | KAMPUNG PASIR LEBAR TIMUR |
| image_090 | TAN CHEE BOON | TAMAN KENARI INDAH |
| image_014 | TUNG KEAN HONG | TAMAN PERAK INDAH |
| image_073 | CHEAH CHEE KHEONG | TAMAN LAPANGAN PERMAI |
| image_105 | DEVAN | JALAN TEMEKONG |

Polanya sistematis: **baseline selalu mengambil baris alamat sebagai nama**, karena heuristiknya
"ambil baris huruf terpanjang", dan di MyKad baris terpanjang memang selalu alamat.

Dan di setiap kasus di atas, **nama yang benar sudah ada utuh sebagai satu block OCR**. Jadi OCR-nya
tidak pernah salah membaca. Pemilihannya yang salah.

### Bukti kuantitatif (tutup dengan ini)

Metodenya: dari output OCR yang sudah ada, hitung CER terbaik yang mungkin dicapai seandainya
pemilihan barisnya sempurna. Caranya brute force, coba semua kombinasi baris yang bersebelahan,
ambil jarak edit terkecil. Tanpa model apapun, deterministik, bisa direproduksi.

| Field | Oracle CER | CER aktual saat itu | Porsi error dari salah pilih |
|---|---|---|---|
| **name** | **5.7 / 8.9** | 50.2 / 54.7 | ~89% |

Satu kalimat penutup: *"Pola ini sistematis, bukan anekdot. Kalau baris yang benar selalu terpilih,
CER nama turun ke 5.7%."*

### Dua catatan jujur untuk sesi tanya jawab

- **birth_date** tidak bisa diukur dengan oracle, karena tanggalnya *diturunkan* dari nomor ID, bukan
  disalin dari teks. Ukuran yang tepat: tanggal bisa direkonstruksi pada 93.9% / 95.8% record, dan
  model akhir mencapai persis 93.9% / 95.8%. **Sudah mentok di ceiling OCR.**
- **address** juga tidak bisa, dan alasannya penting: alamat utuh hanya tertangkap OCR pada
  6.1% / 2.1% record, rata rata cuma 53% token-nya yang muncul. Jadi alamat **terbatas oleh OCR**,
  bukan oleh parser. Ini yang menjelaskan kenapa nanti CER alamat turun jauh tapi exact match-nya
  tetap rendah.

---

## Slide 3: Structure-aware pipeline

**Pesan utama:** Kalau masalahnya memilih baris, bangun sistem yang memang dirancang untuk memilih
baris.

**Langkah 1, classifier per baris.** Tiap baris OCR diklasifikasi (nama / tanggal / alamat / bukan
apa apa) dari fitur geometris dan bentuk teks: posisi relatif, urutan baca, tinggi, rasio digit,
rasio huruf kapital. Label diturunkan otomatis dari ground truth, nol anotasi manual.

**Langkah 2, lapisan aturan struktural.** Eksklusivitas antar field, pengecualian boilerplate kartu,
penjagaan partikel nama (`BIN`, `BINTI`, `A/L`), dan clustering baris bertetangga untuk alamat
multi baris.

### Dampaknya, dipisah per langkah

**name**

| Tahap | CER | WER | Exact |
|---|---|---|---|
| Zero-shot | 50.2 / 54.7 | 63.4 / 73.4 | 20.4 / 16.7 |
| + classifier per baris | 36.6 / 42.2 | 55.5 / 60.7 | 24.5 / 25.0 |
| + lapisan struktural | **20.6 / 29.0** | **42.0 / 43.5** | **38.8 / 31.2** |

**address**

| Tahap | CER | WER | Exact |
|---|---|---|---|
| Zero-shot | 77.4 / 79.9 | 84.7 / 85.8 | 0.0 / 0.0 |
| + classifier per baris | 47.1 / 51.1 | 64.3 / 66.0 | 2.0 / 0.0 |
| + lapisan struktural | **27.1 / 28.5** | **54.9 / 57.3** | **6.1 / 2.1** |

**Talking point:** Lapisan struktural memberi lompatan lebih besar daripada classifier-nya sendiri,
persis seperti yang diprediksi diagnosa di Slide 2.

Detail bagus untuk tanya jawab: oracle 5.7% hanya tercapai kalau boleh menggabungkan **dua** baris.
Dengan satu baris saja mentok di 23%. Itulah kenapa pipeline punya logika penyambung nama yang
terpotong oleh partikel seperti `BIN` dan `BINTI`.

### Before/after kualitatif (lanjutan dari Slide 2)

| Gambar | Ground truth | Baseline0 | Pipeline akhir |
|---|---|---|---|
| image_064 | AFIZAN BIN ZAINAL ABIDIN | KAMPUNG PASIR LEBAR TIMUR | ✅ AFIZAN BIN ZAINAL ABIDIN |
| image_090 | TAN CHEE BOON | TAMAN KENARI INDAH | ✅ TAN CHEE BOON |
| image_014 | TUNG KEAN HONG | TAMAN PERAK INDAH | ✅ TUNG KEAN HONG |
| image_073 | CHEAH CHEE KHEONG | TAMAN LAPANGAN PERMAI | ✅ CHEAH CHEE KHEONG |
| image_105 | DEVAN | JALAN TEMEKONG | ✅ DEVAN |

---

# BAGIAN 2: FINAL RESULT

## Slide 4: The result of modeling with new pipeline

Konfigurasi akhir: **foto mentah → PaddleOCR → classifier per baris → lapisan struktural**.
Tanpa preprocessing gambar. Khusus birth_date ditambah fusi hasil EasyOCR.

| Field | CER | WER | Exact match |
|---|---|---|---|
| birth_date | **4.3 / 4.2** | **4.8 / 4.2** | **93.9 / 95.8** |
| name | 20.6 / 29.0 | 42.0 / 43.5 | 38.8 / 31.2 |
| address | 27.1 / 28.5 | 54.9 / 57.3 | 6.1 / 2.1 |

**Talking point:** Fusi dua engine OCR khusus birth_date menaikkan exact match dari 89.8 ke 93.9.
Fusi yang sama justru **merugikan** nama dan alamat, karena menambah baris teks menggeser fitur
posisi relatif. Jadi kami terapkan secara bedah, hanya di field yang diuntungkan.

---

## Slide 5: The improvements

Dari zero-shot ke pipeline akhir, split yang sama.

| Field | CER | WER | Exact match |
|---|---|---|---|
| name | 50.2 → **20.6** / 54.7 → **29.0** | 63.4 → 42.0 / 73.4 → 43.5 | 20.4 → 38.8 / 16.7 → 31.2 |
| address | 77.4 → **27.1** / 79.9 → **28.5** | 84.7 → 54.9 / 85.8 → 57.3 | 0.0 → 6.1 / 0.0 → 2.1 |
| birth_date | 6.9 → **4.3** / 8.3 → **4.2** | 7.5 → 4.8 / 8.3 → 4.2 | 89.8 → 93.9 / 91.7 → 95.8 |

Penurunan CER relatif: **name −59% / −47%**, **address −65% / −64%**, **birth_date −38% / −49%**.

### Yang diuji dan gagal, tetap kami catat

| Ide | Kenapa gagal |
|---|---|
| Preprocessing gambar | Rata rata netral atau merugikan, OCR memang sudah cukup baik |
| Kamus partikel nama sebagai fitur | Terlalu jarang muncul di training, koefisiennya tidak stabil |
| Skor frekuensi boilerplate sebagai fitur | Model kesulitan belajar dari data per kelas yang tipis |
| Skor n-gram karakter | Masalah yang sama, statistiknya terlalu sedikit |
| Fitur jarak ke wajah terdeteksi | Tidak membantu maupun merugikan |
| Voting 5 model classifier | Alamat jadi lebih buruk karena kalah suara |
| Fusi dua engine OCR untuk nama/alamat | Menambah baris menggeser fitur posisi relatif |

### Perbandingan jenis classifier (bahan bagus untuk tanya jawab)

Empat jenis gradient boosting juga diuji, dan hasilnya berbalik tergantung ada tidaknya lapisan
struktural. CER, val / test.

**Tanpa lapisan struktural**, gradient boosting memang unggul untuk nama:

| Model | CER nama | CER alamat |
|---|---|---|
| Logistic regression | 36.6 / 42.2 | **47.1 / 51.1** |
| HistGradientBoosting | 35.0 / 37.2 | 55.5 / 60.2 |
| LightGBM | 40.7 / 39.0 | 55.1 / 59.8 |
| XGBoost | 32.8 / 36.0 | 53.1 / 57.3 |
| CatBoost | **32.2 / 32.9** | 52.5 / 53.4 |

**Dengan lapisan struktural**, urutannya berbalik dan logistic regression menang di semua field:

| Model | CER nama | CER alamat |
|---|---|---|
| Logistic regression | **20.6 / 29.0** | **27.1 / 28.5** |
| HistGradientBoosting | 27.0 / 31.7 | 51.3 / 57.8 |
| LightGBM | 31.8 / 35.6 | 50.4 / 56.3 |
| XGBoost | 24.7 / 29.6 | 48.9 / 56.3 |
| CatBoost | 22.1 / 28.6 | 45.3 / 46.2 |

**Talking point:** Ini temuan yang menarik. Model yang lebih canggih menang saat dipakai sendirian,
tapi kalah begitu lapisan struktural ditambahkan. Dugaan kami, probabilitas logistic regression lebih
terkalibrasi, sehingga ambang batas di lapisan struktural bekerja lebih andal di atasnya. Kesimpulan
praktisnya: pada data sebesar ini, memperbaiki struktur lebih berharga daripada mengganti model.

---

## Slide 6: Key takeaways

**1. Insight struktur dokumen mengalahkan usaha generik.**
Satu temuan EDA (tanggal lahir tertanam di nomor ID) menyelesaikan satu field penuh dengan satu
aturan regex, mencapai 89.8% exact match tanpa training sama sekali. Tidak ada model yang bisa
menandingi rasio biaya manfaat itu.

**2. Ukur dulu di mana errornya, jangan optimasi yang kelihatan obvious.**
Kami hampir menghabiskan waktu memperbaiki gambar. Setelah diukur ternyata datar, bahkan exact match
turun. Diagnosa kemudian menunjukkan hampir 90% error nama adalah salah memilih baris, bukan salah
membaca. Perbaikan terbesar datang dari sana.

**3. Kami tahu batas hasil kami, dan di mana letaknya.**

- **Alamat terbatas oleh OCR, bukan oleh model.** Alamat utuh hanya tertangkap pada 6.1% / 2.1%
  gambar, dan exact match kami persis di angka itu. Memperbaiki selection tidak akan menolong lagi.
- **Eval set kami 93% MyKad**, karena aturan split mengirim identitas dengan banyak foto ulang ke
  train. Jadi angka di atas paling tepat dibaca sebagai performa pada MyKad.
- **Selisih val-test pada nama** (CER 20.6 vs 29.0) wajar mengingat test set hanya 48 baris.

Sisi positifnya justru kuat: model kami **dilatih pada 97.8% dokumen non-Malaysia** lalu dievaluasi
pada 93% MyKad, dan tetap bekerja. Fitur strukturalnya memang transfer lintas jenis dokumen.

### Contoh yang masih gagal, untuk kejujuran

| Gambar | Ground truth | Pipeline akhir |
|---|---|---|
| image_056 | GRACE LEE SHI JIA | PARKFIELD RESIDENCES |
| image_031 | HAIROL BIN SAILI | KUCHING BRANCH |

Keduanya dokumen dengan kop surat perusahaan yang mengecoh. Menampilkan kasus yang masih gagal
setelah menampilkan yang berhasil membuat kemenangan sebelumnya terbaca jauh lebih kredibel.

---

## Catatan penyampaian

1. **Slide 1b dan Slide 2 adalah pasangan.** Jangan dipisah. Kegagalan preprocessing di 1b yang
   memberi makna pada pertanyaan di Slide 2.
2. **Slide 2 pimpin dengan contoh, tutup dengan angka.** Contohnya menunjukkan mekanisme, angkanya
   menutup tuduhan cherry-picking.
3. **Poin 3 di takeaways jangan dihapus** meski terdengar melemahkan. Juri hampir pasti menanyakan
   komposisi eval set, dan jauh lebih baik Anda yang menyebutnya lebih dulu.
4. Kalau Slide 1 terlalu padat, pecah jadi dua. Tapi kekuatannya justru saat baseline dan kegagalan
   preprocessing terlihat bersebelahan.

## Sumber angka

Semua metrik dibaca dari `reports/experiments.jsonl`. Oracle CER, ablasi MyKad, tingkat
keterpulihan teks, dan contoh kualitatif dihitung ulang dari `reports/paddleocr_{val,test}.json`.
