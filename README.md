# 学習済み画像圧縮モデル - CompressAI 版

このプロジェクトは、**訓練済みの**深層学習画像圧縮モデルを使用して、実際の画像圧縮・展開を体験できる Jupyter ノートブックです。

## 📋 概要

- **論文**: "Joint Autoregressive and Hierarchical Priors for Learned Image Compression" (NeurIPS 2018)
- **実装**: [CompressAI](https://github.com/InterDigitalInc/CompressAI) ライブラリの訓練済みモデル
- **特徴**: 実際に動作する画像圧縮・品質評価・ビットレート比較

## 🚀 クイックスタート

### Google Colab で即座に実行

**🎯 シンプルデモ（推奨）** - 最も簡単な 1 つのアウトプット：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/simple_compression_demo.ipynb)

**🔬 詳細デモ** - 完全な比較・分析版：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/compressai_test.ipynb)

**直接 URL**:

- シンプル版: https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/simple_compression_demo.ipynb
- 詳細版: https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/compressai_test.ipynb

## 📁 ファイル構成

```
compressai-image-compression/
├── simple_compression_demo.ipynb  # 🎯 シンプルデモ（推奨）
├── compressai_test.ipynb          # 🔬 詳細デモ
├── sync.sh                        # GitHub同期スクリプト
├── model.py                       # モデル定義
├── main.py                        # 訓練スクリプト
├── utils.py                       # ユーティリティ関数
├── images/                        # 訓練結果画像
└── README.md                     # このファイル
```

## 💻 ローカル開発ワークフロー

### 前提条件

- Git
- Python 3.7+
- お好みのエディタ（VS Code、Cursor 等）

### セットアップ

1. **リポジトリのクローン**

```bash
git clone https://github.com/Uttchii/learned-image-compression.git
cd learned-image-compression
```

2. **ノートブックの編集**

```bash
# VS Code で編集
code compressai_test.ipynb

# または Cursor で編集
cursor compressai_test.ipynb
```

3. **変更を GitHub に同期**

```bash
./sync.sh
```

4. **Google Colab で実行**

- [ブックマーク用 URL](https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/compressai_test.ipynb) から開く

## 🎯 ノートブックの内容

### 実行可能な機能

- ✅ **訓練済みモデルの自動ダウンロード**: mbt2018 モデル（品質レベル 1-8）
- ✅ **実際の画像圧縮・展開**: テスト画像と実画像の両方
- ✅ **品質評価**: PSNR、圧縮率、ビット/ピクセル
- ✅ **視覚的比較**: 元画像・復元画像・差分画像の並列表示
- ✅ **品質レベル選択**: 1-8 の品質レベル比較
- ✅ **インターネット画像実験**: URL から画像をダウンロードして圧縮

### セル構成

1. **環境セットアップ**: CompressAI のインストールとライブラリインポート
2. **デバイス確認**: GPU/CPU 環境の自動検出
3. **モデル読み込み**: 訓練済みモデルの自動ダウンロード（初回のみ）
4. **テスト画像圧縮**: カラフルなテスト画像での実験
5. **結果可視化**: 圧縮前後の比較と品質評価
6. **実画像実験**: Wikipedia 等からの実画像での圧縮テスト

## 📊 期待される結果

### 典型的な性能（品質レベル 3）

- **PSNR**: 25-35 dB
- **圧縮率**: 10-50 倍
- **ビット/ピクセル**: 0.1-1.0 bpp
- **実行時間**: 数秒〜数分（画像サイズとデバイスによる）

### 品質レベル

- **1**: 最低品質・最高圧縮率（0.1 bpp 程度）
- **3**: バランス型（推奨）（0.3 bpp 程度）
- **5**: 高品質・中圧縮率（0.7 bpp 程度）
- **8**: 最高品質・低圧縮率（1.5 bpp 程度）

## 🔧 開発者向け情報

### 同期スクリプト（sync.sh）

```bash
#!/bin/bash
# 変更をGitHubに自動プッシュ
git add .
git commit -m "Update notebook $(date '+%Y-%m-%d %H:%M:%S')"
git push
echo "✅ 完了！"
echo "🌐 Colab: https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/compressai_test.ipynb"
```

### 便利なエイリアス

`~/.zshrc` に追加すると便利：

```bash
# CompressAI作業用エイリアス
alias colab-sync='cd /path/to/compressai-image-compression && ./sync.sh'
alias colab-edit='cd /path/to/compressai-image-compression && code compressai_test.ipynb'
alias colab-open='open "https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/compressai_test.ipynb"'
```

## 🆚 従来手法との比較

### このプロジェクト（CompressAI）vs 元のリポジトリ

| 項目           | CompressAI 版（このプロジェクト） | 元のリポジトリ     |
| -------------- | --------------------------------- | ------------------ |
| 訓練済みモデル | ✅ 利用可能（8 品質レベル）       | ❌ なし            |
| 実行時間       | 数分で完了                        | 数日間の訓練が必要 |
| 実用性         | すぐに画像圧縮を体験可能          | 研究・開発用       |
| 設定の簡単さ   | pip install のみ                  | 複雑な環境構築     |
| GPU 要件       | CPU 可・GPU 推奨                  | GPU 必須           |

## 🔗 関連リンク

### 技術資料

- **CompressAI**: https://github.com/InterDigitalInc/CompressAI
- **論文 (NeurIPS 2018)**: https://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
- **CompressAI Documentation**: https://interdigitalinc.github.io/CompressAI/
- **Model Zoo**: https://interdigitalinc.github.io/CompressAI/zoo.html

### このプロジェクト

- **GitHub**: https://github.com/Uttchii/learned-image-compression
- **🎯 Google Colab（シンプル）**: https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/simple_compression_demo.ipynb
- **🔬 Google Colab（詳細）**: https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/compressai_test.ipynb

## 🚧 今後の拡張予定

- [ ] 複数の圧縮アルゴリズムとの比較（JPEG、WebP 等）
- [ ] カスタム画像のアップロード機能
- [ ] 動画圧縮への拡張
- [ ] より詳細な品質評価指標の追加（SSIM、MS-SSIM 等）
- [ ] リアルタイム圧縮デモ
- [ ] 圧縮パラメータの可視化

## 🤝 貢献

プルリクエストや課題報告を歓迎します！

### 貢献方法

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは教育・研究目的で作成されています。CompressAI ライブラリのライセンスに従います。

## 🙏 謝辞

- **CompressAI チーム**: 素晴らしいライブラリとモデルの提供
- **論文著者**: Johannes Ballé et al. の研究成果
- **PyTorch コミュニティ**: 深層学習フレームワークの提供

---

**🎉 今すぐ試してみよう！**

🎯 **シンプル版（推奨）**:  
[Google Colab で開く](https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/simple_compression_demo.ipynb)

🔬 **詳細版**:  
[Google Colab で開く](https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/compressai_test.ipynb)

**💡 ヒント**: Google Colab で実行する際は、ランタイムタイプを「GPU」に設定すると高速化されます！
