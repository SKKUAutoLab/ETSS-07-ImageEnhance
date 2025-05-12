<!--# [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/xxxxxx) -->

[English](README.md) | [简体中文](README_cn.md) | 日本語 | [English Blog](src/zoo/dfine/blog.md) | [中文博客](src/zoo/dfine/blog_cn.md)

<h2 align="center">
  D-FINE: Redefine Regression Task of DETRs as Fine&#8209;grained&nbsp;Distribution&nbsp;Refinement
</h2>



<p align="center">
    <a href="https://github.com/Peterande/D-FINE/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://github.com/Peterande/D-FINE/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/Peterande/D-FINE">
    </a>
    <a href="https://github.com/Peterande/D-FINE/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/Peterande/D-FINE?color=olive">
    </a>
    <a href="https://arxiv.org/abs/2410.13842">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2410.13842-red">
    </a>
<!--     <a href="mailto: pengyansong@mail.ustc.edu.cn">
        <img alt="email" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a> -->
      <a href="https://results.pre-commit.ci/latest/github/Peterande/D-FINE/master">
        <img alt="pre-commit.ci status" src="https://results.pre-commit.ci/badge/github/Peterande/D-FINE/master.svg">
    </a>
    <a href="https://github.com/Peterande/D-FINE">
        <img alt="stars" src="https://img.shields.io/github/stars/Peterande/D-FINE">
    </a>
</p>



<p align="center">
    📄 これは論文の公式実装です:
    <br>
    <a href="https://arxiv.org/abs/2410.13842">D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement</a>
  </p>
<p align="center">
  D-FINE: DETRの回帰タスクを細粒度分布最適化として再定義
</p>



<p align="center">
Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun, and Feng Wu
</p>

<p align="center">
中国科学技術大学
</p>

<p align="center">
    <a href="https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=d-fine-redefine-regression-task-in-detrs-as">
        <img alt="sota" src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/d-fine-redefine-regression-task-in-detrs-as/real-time-object-detection-on-coco">
    </a>
</p>

<!-- <table><tr>
<td><img src=https://github.com/Peterande/storage/blob/master/latency.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/master/params.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/master/flops.png border=0 width=333></td>
</tr></table> -->

<p align="center">
<strong>もしD-FINEが気に入ったら、ぜひ⭐をください！あなたのサポートが私たちのモチベーションになります！</strong>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/Peterande/storage/master/figs/stats_padded.png" width="1000">
</p>

D-FINEは、DETRの境界ボックス回帰タスクを細粒度分布最適化（FDR）として再定義し、グローバル最適な位置特定自己蒸留（GO-LSD）を導入することで、追加の推論およびトレーニングコストを増やすことなく、優れたパフォーマンスを実現する強力なリアルタイムオブジェクト検出器です。

<details open>
<summary> ビデオ </summary>

D-FINEとYOLO11を使用して、[YouTube](https://www.youtube.com/watch?v=CfhEWj9sd9A)の複雑な街並みのビデオでオブジェクト検出を行いました。逆光、モーションブラー、密集した群衆などの厳しい条件にもかかわらず、D-FINE-Xはほぼすべてのターゲットを検出し、バックパック、自転車、信号機などの微妙な小さなオブジェクトも含まれます。その信頼スコアとぼやけたエッジの位置特定精度はYOLO11よりもはるかに高いです。

<!-- We use D-FINE and YOLO11 on a street scene video from [YouTube](https://www.youtube.com/watch?v=CfhEWj9sd9A). Despite challenges like backlighting, motion blur, and dense crowds, D-FINE-X outperforms YOLO11x, detecting more objects with higher confidence and better precision. -->

https://github.com/user-attachments/assets/e5933d8e-3c8a-400e-870b-4e452f5321d9

</details>

## 🚀 更新情報
- [x] **\[2024.10.18\]** D-FINEシリーズをリリース。
- [x] **\[2024.10.25\]** カスタムデータセットの微調整設定を追加 ([#7](https://github.com/Peterande/D-FINE/issues/7))。
- [x] **\[2024.10.30\]** D-FINE-L (E25) 事前トレーニングモデルを更新し、パフォーマンスが2.0%向上。
- [x] **\[2024.11.07\]** **D-FINE-N** をリリース, COCO で 42.8% の AP<sup>val</sup> を達成 @ 472 FPS<sup>T4</sup>!

## モデルズー

### COCO
| モデル | データセット | AP<sup>val</sup> | パラメータ数 | レイテンシ | GFLOPs | config | checkpoint | logs |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
**D&#8209;FINE&#8209;N** | COCO | **42.8** | 4M | 2.12ms | 7 | [yml](options/dfine/dfine_hgnetv2_n_coco.yml) | [42.8](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/coco/dfine_n_coco_log.txt)
**D&#8209;FINE&#8209;S** | COCO | **48.5** | 10M | 3.49ms | 25 | [yml](options/dfine/dfine_hgnetv2_s_coco.yml) | [48.5](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/coco/dfine_s_coco_log.txt)
**D&#8209;FINE&#8209;M** | COCO | **52.3** | 19M | 5.62ms | 57 | [yml](options/dfine/dfine_hgnetv2_m_coco.yml) | [52.3](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/coco/dfine_m_coco_log.txt)
**D&#8209;FINE&#8209;L** | COCO | **54.0** | 31M | 8.07ms | 91 | [yml](options/dfine/dfine_hgnetv2_l_coco.yml) | [54.0](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/coco/dfine_l_coco_log.txt)
**D&#8209;FINE&#8209;X** | COCO | **55.8** | 62M | 12.89ms | 202 | [yml](options/dfine/dfine_hgnetv2_x_coco.yml) | [55.8](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/coco/dfine_x_coco_log.txt)


### Objects365+COCO
| モデル | データセット | AP<sup>val</sup> | パラメータ数 | レイテンシ | GFLOPs | config | checkpoint | logs |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
**D&#8209;FINE&#8209;S** | Objects365+COCO | **50.7** | 10M | 3.49ms | 25 | [yml](options/dfine/objects365/dfine_hgnetv2_s_obj2coco.yaml) | [50.7](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/obj2coco/dfine_s_obj2coco_log.txt)
**D&#8209;FINE&#8209;M** | Objects365+COCO | **55.1** | 19M | 5.62ms | 57 | [yml](options/dfine/objects365/dfine_hgnetv2_m_obj2coco.yaml) | [55.1](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/obj2coco/dfine_m_obj2coco_log.txt)
**D&#8209;FINE&#8209;L** | Objects365+COCO | **57.3** | 31M | 8.07ms | 91 | [yml](options/dfine/objects365/dfine_hgnetv2_l_obj2coco.yaml) | [57.3](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/obj2coco/dfine_l_obj2coco_log_e25.txt)
**D&#8209;FINE&#8209;X** | Objects365+COCO | **59.3** | 62M | 12.89ms | 202 | [yml](options/dfine/objects365/dfine_hgnetv2_x_obj2coco.yaml) | [59.3](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/obj2coco/dfine_x_obj2coco_log.txt)

**微調整のために Objects365 の事前学習モデルを使用することを強くお勧めします：**

⚠️ 重要なお知らせ：このプリトレインモデルは複雑なシーンの理解に有益ですが、カテゴリが非常に単純な場合、過学習や最適ではない性能につながる可能性がありますので、ご注意ください。

<details> <summary><strong> 🔥 Objects365で事前トレーニングされたモデル（最良の汎化性能）</strong></summary>


| モデル | データセット | AP<sup>val</sup> | AP<sup>5000</sup> | パラメータ数 | レイテンシ | GFLOPs | config | checkpoint | logs |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: |
**D&#8209;FINE&#8209;S** | Objects365 | **31.0** | **30.5** | 10M | 3.49ms | 25 | [yml](options/dfine/objects365/dfine_hgnetv2_s_obj365.yaml) | [30.5](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj365.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/obj365/dfine_s_obj365_log.txt)
**D&#8209;FINE&#8209;M** | Objects365 | **38.6** | **37.4** | 19M | 5.62ms | 57 | [yml](options/dfine/objects365/dfine_hgnetv2_m_obj365.yaml) | [37.4](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj365.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/obj365/dfine_m_obj365_log.txt)
**D&#8209;FINE&#8209;L** | Objects365 | - | **40.6** | 31M | 8.07ms | 91 | [yml](options/dfine/objects365/dfine_hgnetv2_l_obj365.yaml) | [40.6](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj365.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/obj365/dfine_l_obj365_log.txt)
**D&#8209;FINE&#8209;L (E25)** | Objects365 | **44.7** | **42.6** | 31M | 8.07ms | 91 | [yml](options/dfine/objects365/dfine_hgnetv2_l_obj365.yaml) | [42.6](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj365_e25.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/obj365/dfine_l_obj365_log_e25.txt)
**D&#8209;FINE&#8209;X** | Objects365 | **49.5** | **46.5** | 62M | 12.89ms | 202 | [yml](options/dfine/objects365/dfine_hgnetv2_x_obj365.yaml) | [46.5](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj365.pth) | [url](https://raw.githubusercontent.com/Peterande/storage/refs/heads/master/logs/obj365/dfine_x_obj365_log.txt)
- **E25**: 再トレーニングし、事前トレーニングを25エポックに延長。
- **AP<sup>val</sup>** は *Objects365* のフルバリデーションセットで評価されます。
- **AP<sup>5000</sup>** は *Objects365* 検証セットの最初の5000サンプルで評価されます。
</details>

**注意事項:**
- **AP<sup>val</sup>** は *MSCOCO val2017* データセットで評価されます。
- **レイテンシ** は単一のT4 GPUで $batch\\_size = 1$, $fp16$, および $TensorRT==10.4.0$ で評価されます。
- **Objects365+COCO** は *Objects365* で事前トレーニングされた重みを使用して *COCO* で微調整されたモデルを意味します。



## クイックスタート

### セットアップ

```shell
conda create -n dfine python=3.11.9
conda activate dfine
pip install -r requirements.txt
```


### データ準備

<details>
<summary> COCO2017 データセット </summary>

1. [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) または [COCO](https://cocodataset.org/#download) からCOCO2017をダウンロードします。
1. [coco_detection.yml](options/dataset/coco_detection.yaml) のパスを修正します。

    ```yaml
    train_dataloader:
        img_folder: /data/COCO2017/train2017/
        ann_file: /data/COCO2017/annotations/instances_train2017.json
    val_dataloader:
        img_folder: /data/COCO2017/val2017/
        ann_file: /data/COCO2017/annotations/instances_val2017.json
    ```

</details>

<details>
<summary> Objects365 データセット </summary>

1. [OpenDataLab](https://opendatalab.com/OpenDataLab/Objects365) からObjects365をダウンロードします。

2. ベースディレクトリを設定します：
```shell
export BASE_DIR=/data/Objects365/data
```

3. ダウンロードしたファイルを解凍し、以下のディレクトリ構造に整理します：

```shell
${BASE_DIR}/train
├── images
│   ├── v1
│   │   ├── patch0
│   │   │   ├── 000000000.jpg
│   │   │   ├── 000000001.jpg
│   │   │   └── ... (more images)
│   ├── v2
│   │   ├── patchx
│   │   │   ├── 000000000.jpg
│   │   │   ├── 000000001.jpg
│   │   │   └── ... (more images)
├── zhiyuan_objv2_train.json
```

```shell
${BASE_DIR}/val
├── images
│   ├── v1
│   │   ├── patch0
│   │   │   ├── 000000000.jpg
│   │   │   └── ... (more images)
│   ├── v2
│   │   ├── patchx
│   │   │   ├── 000000000.jpg
│   │   │   └── ... (more images)
├── zhiyuan_objv2_val.json
```

4. 検証セットの画像を保存する新しいディレクトリを作成します：
```shell
mkdir -p ${BASE_DIR}/train/images_from_val
```

5. valディレクトリのv1およびv2フォルダをtrain/images_from_valディレクトリにコピーします
```shell
cp -r ${BASE_DIR}/val/images/v1 ${BASE_DIR}/train/images_from_val/
cp -r ${BASE_DIR}/val/images/v2 ${BASE_DIR}/train/images_from_val/
```

6. remap_obj365.pyを実行して、検証セットの一部をトレーニングセットにマージします。具体的には、このスクリプトはインデックスが5000から800000のサンプルを検証セットからトレーニングセットに移動します。
```shell
python tools/remap_obj365.py --base_dir ${BASE_DIR}
```


7. resize_obj365.pyスクリプトを実行して、データセット内の最大エッジ長が640ピクセルを超える画像をリサイズします。ステップ5で生成された更新されたJSONファイルを使用してサンプルデータを処理します。トレーニングセットと検証セットの両方の画像をリサイズして、一貫性を保ちます。
```shell
python tools/resize_obj365.py --base_dir ${BASE_DIR}
```

8. [obj365_detection.yml](options/dataset/obj365_detection.yaml) のパスを修正します。

    ```yaml
    train_dataloader:
        img_folder: /data/Objects365/data/train
        ann_file: /data/Objects365/data/train/new_zhiyuan_objv2_train_resized.json
    val_dataloader:
        img_folder: /data/Objects365/data/val/
        ann_file: /data/Objects365/data/val/new_zhiyuan_objv2_val_resized.json
    ```


</details>

<details>
<summary>CrowdHuman</summary>

こちらからCOCOフォーマットのデータセットをダウンロードしてください：[リンク](https://aistudio.baidu.com/datasetdetail/231455)

</details>

<details>
<summary>カスタムデータセット</summary>

カスタムデータセットでトレーニングするには、COCO形式で整理する必要があります。以下の手順に従ってデータセットを準備してください：

1. **`remap_mscoco_category` を `False` に設定します**：

    これにより、カテゴリIDがMSCOCOカテゴリに自動的にマッピングされるのを防ぎます。

    ```yaml
    remap_mscoco_category: False
    ```

2. **画像を整理します**：

    データセットディレクトリを以下のように構造化します：

    ```shell
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── ...
    ```

    - **`images/train/`**: すべてのトレーニング画像を含みます。
    - **`images/val/`**: すべての検証画像を含みます。
    - **`annotations/`**: COCO形式の注釈ファイルを含みます。

3. **注釈をCOCO形式に変換します**：

    注釈がまだCOCO形式でない場合は、変換する必要があります。以下のPythonスクリプトを参考にするか、既存のツールを利用してください：

    ```python
    import json

    def convert_to_coco(input_annotations, output_annotations):
        # 変換ロジックをここに実装します
        pass

    if __name__ == "__main__":
        convert_to_coco('path/to/your_annotations.json', 'dataset/annotations/instances_train.json')
    ```

4. **設定ファイルを更新します**：

    [custom_detection.yml](options/dataset/custom_detection.yaml) を修正します。

    ```yaml
    task: detection

    evaluator:
      type: CocoEvaluator
      iou_types: ['bbox', ]

    num_classes: 777 # データセットのクラス数
    remap_mscoco_category: False

    train_dataloader:
      type: DataLoader
      dataset:
        type: CocoDetection
        img_folder: /data/yourdataset/train
        ann_file: /data/yourdataset/train/train.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~
      shuffle: True
      num_workers: 4
      drop_last: True
      collate_fn:
        type: BatchImageCollateFunction

    val_dataloader:
      type: DataLoader
      dataset:
        type: CocoDetection
        img_folder: /data/yourdataset/val
        ann_file: /data/yourdataset/val/ann.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~
      shuffle: False
      num_workers: 4
      drop_last: False
      collate_fn:
        type: BatchImageCollateFunction
    ```

</details>


## 使用方法
<details open>
<summary> COCO2017 </summary>

<!-- <summary>1. トレーニング </summary> -->
1. モデルを設定します
```shell
export model=l  # n s m l x
```

2. トレーニング
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --use-amp --seed=0
```

<!-- <summary>2. テスト </summary> -->
3. テスト
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --test-only -r model.pth
```

<!-- <summary>3. 微調整 </summary> -->
4. 微調整
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --use-amp --seed=0 -t model.pth
```
</details>


<details>
<summary> Objects365からCOCO2017へ </summary>

1. モデルを設定します
```shell
export model=l  # n s m l x
```

2. Objects365でトレーニング
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj365.yml --use-amp --seed=0
```

3. COCO2017で微調整
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj2coco.yml --use-amp --seed=0 -t model.pth
```

<!-- <summary>2. テスト </summary> -->
4. テスト
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --test-only -r model.pth
```
</details>


<details>
<summary> カスタムデータセット </summary>

1. モデルを設定します
```shell
export model=l  # n s m l x
```

2. カスタムデータセットでトレーニング
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --use-amp --seed=0
```
<!-- <summary>2. テスト </summary> -->
3. テスト
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --test-only -r model.pth
```

4. カスタムデータセットで微調整
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/custom/objects365/dfine_hgnetv2_${model}_obj2custom.yml --use-amp --seed=0 -t model.pth
```

5. **[オプション]** クラスマッピングを変更します：

Objects365の事前トレーニング済みの重みを使用してカスタムデータセットでトレーニングする場合、例ではデータセットに `'Person'` と `'Car'` クラスのみが含まれていると仮定しています。特定のタスクに対して収束を早めるために、`src/solver/_solver.py` の `self.obj365_ids` を以下のように変更できます：

```python
self.obj365_ids = [0, 5]  # Person, Cars
```
これらをデータセットの対応するクラスに置き換えることができます。Objects365クラスとその対応IDのリスト：
https://github.com/Peterande/D-FINE/blob/352a94ece291e26e1957df81277bef00fe88a8e3/src/solver/_solver.py#L330

新しいトレーニングコマンド：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --use-amp --seed=0 -t model.pth
```

ただし、クラスマッピングを変更したくない場合、事前トレーニング済みのObjects365の重みは変更なしでそのまま使用できます。クラスマッピングの変更はオプションであり、特定のタスクに対して収束を早める可能性があります。



</details>

<details>
<summary> バッチサイズのカスタマイズ </summary>

例えば、COCO2017でD-FINE-Lをトレーニングする際にバッチサイズを2倍にしたい場合、以下の手順に従ってください：

1. **[dataloader.yml](options/dfine/include/dataloader.yaml) を修正して `total_batch_size` を増やします**：

    ```yaml
    train_dataloader:
        total_batch_size: 64  # 以前は32、今は2倍
    ```

2. **[dfine_hgnetv2_l_coco.yml](options/dfine/dfine_hgnetv2_l_coco.yml) を修正します**。以下のように主要なパラメータを調整します：

    ```yaml
    optimizer:
    type: AdamW
    params:
        -
        params: '^(?=.*backbone)(?!.*norm|bn).*$'
        lr: 0.000025  # 2倍、線形スケーリング法則
        -
        params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
        weight_decay: 0.

    lr: 0.0005  # 2倍、線形スケーリング法則
    betas: [0.9, 0.999]
    weight_decay: 0.0001  # グリッドサーチが必要です

    ema:  # EMA設定を追加
        decay: 0.9998  # 1 - (1 - decay) * 2 によって調整
        warmups: 500  # 半分

    lr_warmup_scheduler:
        warmup_duration: 250  # 半分
    ```

</details>


<details>
<summary> 入力サイズのカスタマイズ </summary>

COCO2017で **D-FINE-L** を320x320の入力サイズでトレーニングしたい場合、以下の手順に従ってください：

1. **[dataloader.yml](options/dfine/include/dataloader.yaml) を修正します**：

    ```yaml

    train_dataloader:
    dataset:
        transforms:
            ops:
                - {type: Resize, size: [320, 320], }
    collate_fn:
        base_size: 320
    dataset:
        transforms:
            ops:
                - {type: Resize, size: [320, 320], }
    ```

2. **[dfine_hgnetv2.yml](options/dfine/include/dfine_hgnetv2.yaml) を修正します**：

    ```yaml
    eval_spatial_size: [320, 320]
    ```

</details>

## ツール
<details>
<summary> デプロイ </summary>

<!-- <summary>4. onnxのエクスポート </summary> -->
1. セットアップ
```shell
pip install onnx onnxsim
export model=l  # n s m l x
```

2. onnxのエクスポート
```shell
python tools/deployment/export_onnx.py --check -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
```

3. [tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) のエクスポート
```shell
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

</details>

<details>
<summary> 推論（可視化） </summary>


1. セットアップ
```shell
pip install -r tools/inference/requirements.txt
export model=l  # n s m l x
```


<!-- <summary>5. 推論 </summary> -->
2. 推論 (onnxruntime / tensorrt / torch)

現在、画像とビデオの推論がサポートされています。
```shell
python tools/inference/onnx_inf.py --onnx model.onnx --input image.jpg  # video.mp4
python tools/inference/trt_inf.py --trt model.engine --input image.jpg
python tools/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --input image.jpg --device cuda:0
```
</details>

<details>
<summary> ベンチマーク </summary>

1. セットアップ
```shell
pip install -r tools/benchmark/requirements.txt
export model=l  # n s m l x
```

<!-- <summary>6. ベンチマーク </summary> -->
2. モデルのFLOPs、MACs、およびパラメータ数
```shell
python tools/benchmark/get_info.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml
```

2. TensorRTのレイテンシ
```shell
python tools/benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```
</details>

<details>
<summary> Fiftyoneの可視化  </summary>

1. セットアップ
```shell
pip install fiftyone
export model=l  # n s m l x
```
4. Voxel51 Fiftyoneの可視化 ([fiftyone](https://github.com/voxel51/fiftyone))
```shell
python tools/visualization/fiftyone_vis.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
```
</details>

<details>
<summary> その他 </summary>

1. 自動再開トレーニング
```shell
bash reference/safe_training.sh
```

2. モデルの重みの変換
```shell
python reference/convert_weight.py model.pth
```
</details>

## 図と可視化

<details>
<summary> FDRとGO-LSD </summary>

1. FDRを搭載したD-FINEの概要。より細粒度の中間表現として機能する確率分布は、残差的にデコーダ層によって逐次最適化されます。
不均一な重み付け関数が適用され、より細かい位置特定が可能になります。

<p align="center">
    <img src="https://raw.githubusercontent.com/Peterande/storage/master/figs/fdr-1.jpg" alt="細粒度分布最適化プロセス" width="1000">
</p>

2. GO-LSDプロセスの概要。最終層の最適化された分布からの位置特定知識は、デカップリングされた重み付け戦略を使用してDDF損失を通じて前の層に蒸留されます。

<p align="center">
    <img src="https://raw.githubusercontent.com/Peterande/storage/master/figs/go_lsd-1.jpg" alt="GO-LSDプロセス" width="1000">
</p>

</details>

<details open>
<summary> 分布 </summary>

初期および最適化された境界ボックスと、未重み付けおよび重み付けされた分布とともに、さまざまな検出シナリオにおけるFDRの可視化。

<p align="center">
    <img src="https://raw.githubusercontent.com/Peterande/storage/master/figs/merged_image.jpg" width="1000">
</p>

</details>

<details>
<summary> 難しいケース </summary>

以下の可視化は、さまざまな複雑な検出シナリオにおけるD-FINEの予測を示しています。これらのシナリオには、遮蔽、低光条件、モーションブラー、被写界深度効果、および密集したシーンが含まれます。これらの課題にもかかわらず、D-FINEは一貫して正確な位置特定結果を生成します。

<p align="center">
    <img src="https://raw.githubusercontent.com/Peterande/storage/master/figs/hard_case-1.jpg" alt="複雑なシナリオにおけるD-FINEの予測" width="1000">
</p>

</details>


<!-- <div style="display: flex; flex-wrap: wrap; justify-content: center; margin: 0; padding: 0;">
    <img src="https://raw.githubusercontent.com/Peterande/storage/master/figs/merged_image.jpg" style="width:99.96%; margin: 0; padding: 0;" />
</div>

<table><tr>
<td><img src=https://raw.githubusercontent.com/Peterande/storage/master/figs/merged_image.jpg border=0 width=1000></td>
</tr></table> -->




## 引用
もし`D-FINE`やその方法をあなたの仕事で使用する場合、以下のBibTeXエントリを引用してください：
<details open>
<summary> bibtex </summary>

```latex
@misc{peng2024dfine,
      title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
      author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
      year={2024},
      eprint={2410.13842},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
</details>

## 謝辞
私たちの仕事は [RT-DETR](https://github.com/lyuwenyu/RT-DETR) に基づいています。
[RT-DETR](https://github.com/lyuwenyu/RT-DETR), [GFocal](https://github.com/implus/GFocal), [LD](https://github.com/HikariTJU/LD), および [YOLOv9](https://github.com/WongKinYiu/yolov9) からのインスピレーションに感謝します。

✨ 貢献を歓迎し、質問があればお気軽にお問い合わせください！ ✨
