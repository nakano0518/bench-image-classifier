## 本アプリケーションの概要

### ベンチ画像分類モデル搭載 WebAPI
ベンチの画像であるかどうかを判定する機械学習モデルを実装した WebAPI

### 作成背景
- [Courseraの機械学習コース (11week) を修了](https://github.com/nakano0518/coursera-machine-learning)したので学習した内容のアウトプット
- BenchMapアプリケーション (web、モバイル) に画像判定のバリデーション導入

### 使い方 (例. POSTMAN)

<img width="930" alt="bench-map-classifier-postman" src="https://user-images.githubusercontent.com/54522567/115045789-cc8d3a00-9f11-11eb-9cd5-e6fe33949bfd.PNG">

#### ⇒ labelの値が1であればベンチの画像、0であればベンチでない画像と判定される


## 機能一覧および使用技術一覧

- ### バックエンド
  - Flask (Python) を使用し、下記の機械学習モデルを組み込み webAPI 化。

- ### 機械学習
  - CNN (畳み込みニューラルネットワーク) のシーケンシャルモデルを採用
  - ベンチ (椅子など座るもの含む) の画像、ベンチでない画像を用意
  - ベンチ画像のラベルを 1、ベンチでない画像のラベルを 0 と設定。
  - トレーニングセットおよびクロスバリデーションセットに分割後、numpy 配列型に変換
  - 正規化したのち、モデルのトレーニングを実行
  - 結果、分類モデルの精度は**75%**

- ### インフラ
  Herokuにデプロイ (無料枠内) 。その際、下記の制約の中で Heroku にデプロイする必要があった。
  - Heroku の Slug サイズ 300MB 制約を下記によりクリア
    - tensorflow のバージョンを調整
  - Heroku のメモリ超過エラー (メモリ制約) を下記によりクリア
    - worker 数を減らす
    - メモリにロードするモデルのサイズを減らす  
      (コンパイルせずモデルを保存し、Heroku 上でモデルをロード、そこでコンパイルをする)
　

## 今後の課題
- 画像のベンチが小さい場合にラベル0と誤判定される問題  
⇒ ベンチが小さく写る画像の量を増やしてトレーニングし改善する予定  
  
  ※BenchMapアプリケーション(web、モバイル)のベンチ画像判定のバリデーションは  
  　現在、GoogleCloudVisionにより対応しているが、  
  　アプリが成熟し、リクエスト数が増えた場合に課金されることが予想されるので、  
  　その時までに、本モデルの改善・実装を行う予定。  