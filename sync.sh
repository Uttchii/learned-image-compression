#!/bin/bash
git add . && git commit -m "Update $(date)" && git push
echo "✅ GitHubに同期完了！"
echo "🌐 Google Colab: https://colab.research.google.com/github/Uttchii/learned-image-compression/blob/main/compressai_test.ipynb"
