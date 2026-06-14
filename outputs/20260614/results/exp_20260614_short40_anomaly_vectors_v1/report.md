# Object-Level Anomaly Vector Training

本实验改为只训练 anomaly vectors。Normal 不再作为显式 prototype；对象异常分数低即判为 normal。

- anomaly_vectors: `8`
- threshold: `0.5`
- selected train/val/openset packages: `32/10/3`

## 本次小范围测试覆盖的异常

这轮 short40 验证不是全量训练，而是从 Avenue、ShanghaiTech、NWPU、UCF-Crime 中抽取短视频 package，覆盖 T01-T05 五个训练异常大类，并保留 R06 作为 open-set 观察样本。样本单位是 `单个 object track_id + 一个短时间窗口 + 该窗口内 bbox 对应的 Qwen3-VL visual tokens`。

| 类别 | 中文含义 | 本轮正样本 object-window | 细分类 | 涉及数据集 | 主要对象类别 |
|---|---|---:|---|---|---|
| T01 | 个体人员行为异常 | 12 | S01 快速奔跑/冲闯；S02 异常姿态/摔倒趴伏 | Avenue, ShanghaiTech | person |
| T02 | 轻型代步工具违规 | 19 | S05 自行车/电动车区域违规 | ShanghaiTech | person, bicycle |
| T03 | 机动车/大型车辆通行违规 | 17 | S09 机动车区域/方向违规；S10 违规停车/异常停靠 | ShanghaiTech, NWPU | person, car, motorcycle, truck, bicycle, skateboard |
| T04 | 冲突攻击与群体秩序异常 | 31 | S13 身体攻击/殴打虐待；S14 群体聚集/秩序异常 | UCF-Crime, ShanghaiTech, NWPU | person, bicycle |
| T05 | 物体操作与财物异常 | 9 | S15 抛掷/乱扔物体；S17 异常携带/搬运/操作物体；S18 财物犯罪 | Avenue, ShanghaiTech | person |
| R06 | 低频高危事件，仅 open-set eval | 4 | S21 动物入侵 | NWPU | dog, bird |

正常对照样本共 `87` 个 object-window，其中 `77` 个来自非异常时间段，`10` 个来自异常时间段内但与异常事件无关的对象。正常对象类别包括 person、backpack、car、bicycle、handbag。也就是说，这轮测试同时验证了“异常对象应该高分”和“正常时间段/无关对象应该低分”。

本轮涉及的异常 package：

- T01: `Avenue_08`, `Avenue_18`, `Avenue_21`, `ShanghaiTech_01_0027`, `ShanghaiTech_01_0055`, `ShanghaiTech_03_0035`, `ShanghaiTech_03_0041`, `ShanghaiTech_07_0007`, `ShanghaiTech_07_0009`
- T02: `ShanghaiTech_01_0014`, `ShanghaiTech_01_0133`, `ShanghaiTech_01_0139`, `ShanghaiTech_01_0162`, `ShanghaiTech_01_0163`, `ShanghaiTech_06_0144`, `ShanghaiTech_06_0145`, `ShanghaiTech_06_0153`, `ShanghaiTech_12_0173`
- T03: `NWPU_D043_02`, `NWPU_D047_06`, `NWPU_D068_02`, `ShanghaiTech_01_0016`, `ShanghaiTech_01_0130`, `ShanghaiTech_01_0132`, `ShanghaiTech_06_0144`, `ShanghaiTech_06_0147`, `ShanghaiTech_12_0148`
- T04: `NWPU_D094_01`, `ShanghaiTech_01_0052`, `ShanghaiTech_03_0033`, `ShanghaiTech_05_0024`, `ShanghaiTech_07_0006`, `ShanghaiTech_07_0048`, `UCF-Crime_Assault_Assault024_x264`, `UCF-Crime_Assault_Assault039_x264`
- T05: `Avenue_11`, `Avenue_20`, `ShanghaiTech_03_0031`, `ShanghaiTech_03_0039`, `ShanghaiTech_05_0017`, `ShanghaiTech_07_0008`, `ShanghaiTech_08_0079`, `ShanghaiTech_09_0057`
- R06: `NWPU_D003_05`, `NWPU_D013_01`, `NWPU_D038_02`

## Package Selection

```json
{
  "train": [
    {
      "package_id": "Avenue_21",
      "dataset": "avenue",
      "frames": 76,
      "labels": [
        "T01"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/avenue/Avenue_21"
    },
    {
      "package_id": "ShanghaiTech_06_0145",
      "dataset": "shanghaitech",
      "frames": 217,
      "labels": [
        "T02"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_06_0145"
    },
    {
      "package_id": "ShanghaiTech_01_0132",
      "dataset": "shanghaitech",
      "frames": 265,
      "labels": [
        "T03"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0132"
    },
    {
      "package_id": "UCF-Crime_Assault_Assault039_x264",
      "dataset": "ucf_crime",
      "frames": 295,
      "labels": [
        "T04"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/ucf_crime/UCF-Crime_Assault_Assault039_x264"
    },
    {
      "package_id": "ShanghaiTech_05_0017",
      "dataset": "shanghaitech",
      "frames": 433,
      "labels": [
        "T05"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_05_0017"
    },
    {
      "package_id": "ShanghaiTech_01_0055",
      "dataset": "shanghaitech",
      "frames": 313,
      "labels": [
        "T01"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0055"
    },
    {
      "package_id": "ShanghaiTech_01_0162",
      "dataset": "shanghaitech",
      "frames": 193,
      "labels": [
        "T02"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0162"
    },
    {
      "package_id": "NWPU_D068_02",
      "dataset": "nwpu",
      "frames": 301,
      "labels": [
        "T03"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/nwpu/NWPU_D068_02"
    },
    {
      "package_id": "ShanghaiTech_01_0052",
      "dataset": "shanghaitech",
      "frames": 337,
      "labels": [
        "T04"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0052"
    },
    {
      "package_id": "ShanghaiTech_09_0057",
      "dataset": "shanghaitech",
      "frames": 361,
      "labels": [
        "T05"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_09_0057"
    },
    {
      "package_id": "Avenue_18",
      "dataset": "avenue",
      "frames": 294,
      "labels": [
        "T01"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/avenue/Avenue_18"
    },
    {
      "package_id": "ShanghaiTech_12_0173",
      "dataset": "shanghaitech",
      "frames": 217,
      "labels": [
        "T02"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_12_0173"
    },
    {
      "package_id": "ShanghaiTech_01_0130",
      "dataset": "shanghaitech",
      "frames": 337,
      "labels": [
        "T03"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0130"
    },
    {
      "package_id": "ShanghaiTech_07_0048",
      "dataset": "shanghaitech",
      "frames": 241,
      "labels": [
        "T04"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_07_0048"
    },
    {
      "package_id": "ShanghaiTech_07_0008",
      "dataset": "shanghaitech",
      "frames": 457,
      "labels": [
        "T05"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_07_0008"
    },
    {
      "package_id": "ShanghaiTech_07_0009",
      "dataset": "shanghaitech",
      "frames": 313,
      "labels": [
        "T01"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_07_0009"
    },
    {
      "package_id": "ShanghaiTech_01_0133",
      "dataset": "shanghaitech",
      "frames": 217,
      "labels": [
        "T02"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0133"
    },
    {
      "package_id": "ShanghaiTech_01_0016",
      "dataset": "shanghaitech",
      "frames": 337,
      "labels": [
        "T03"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0016"
    },
    {
      "package_id": "ShanghaiTech_07_0006",
      "dataset": "shanghaitech",
      "frames": 385,
      "labels": [
        "T04"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_07_0006"
    },
    {
      "package_id": "Avenue_20",
      "dataset": "avenue",
      "frames": 273,
      "labels": [
        "T05"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/avenue/Avenue_20"
    },
    {
      "package_id": "Avenue_08",
      "dataset": "avenue",
      "frames": 36,
      "labels": [
        "T01"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/avenue/Avenue_08"
    },
    {
      "package_id": "ShanghaiTech_06_0153",
      "dataset": "shanghaitech",
      "frames": 217,
      "labels": [
        "T02"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_06_0153"
    },
    {
      "package_id": "ShanghaiTech_12_0148",
      "dataset": "shanghaitech",
      "frames": 313,
      "labels": [
        "T03"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_12_0148"
    },
    {
      "package_id": "NWPU_D094_01",
      "dataset": "nwpu",
      "frames": 401,
      "labels": [
        "T04"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/nwpu/NWPU_D094_01"
    },
    {
      "package_id": "ShanghaiTech_03_0031",
      "dataset": "shanghaitech",
      "frames": 529,
      "labels": [
        "T05"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_03_0031"
    },
    {
      "package_id": "ShanghaiTech_03_0041",
      "dataset": "shanghaitech",
      "frames": 457,
      "labels": [
        "T01"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_03_0041"
    },
    {
      "package_id": "ShanghaiTech_01_0014",
      "dataset": "shanghaitech",
      "frames": 265,
      "labels": [
        "T02"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0014"
    },
    {
      "package_id": "NWPU_D047_06",
      "dataset": "nwpu",
      "frames": 401,
      "labels": [
        "T03"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/nwpu/NWPU_D047_06"
    },
    {
      "package_id": "ShanghaiTech_03_0033",
      "dataset": "shanghaitech",
      "frames": 313,
      "labels": [
        "T04"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_03_0033"
    },
    {
      "package_id": "ShanghaiTech_08_0079",
      "dataset": "shanghaitech",
      "frames": 241,
      "labels": [
        "T05"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_08_0079"
    },
    {
      "package_id": "ShanghaiTech_03_0035",
      "dataset": "shanghaitech",
      "frames": 385,
      "labels": [
        "T01"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_03_0035"
    },
    {
      "package_id": "ShanghaiTech_06_0144",
      "dataset": "shanghaitech",
      "frames": 241,
      "labels": [
        "T02",
        "T03"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_06_0144"
    }
  ],
  "val": [
    {
      "package_id": "ShanghaiTech_01_0027",
      "dataset": "shanghaitech",
      "frames": 409,
      "labels": [
        "T01"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0027"
    },
    {
      "package_id": "ShanghaiTech_01_0163",
      "dataset": "shanghaitech",
      "frames": 265,
      "labels": [
        "T02"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0163"
    },
    {
      "package_id": "ShanghaiTech_06_0147",
      "dataset": "shanghaitech",
      "frames": 265,
      "labels": [
        "T03"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_06_0147"
    },
    {
      "package_id": "ShanghaiTech_05_0024",
      "dataset": "shanghaitech",
      "frames": 433,
      "labels": [
        "T04"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_05_0024"
    },
    {
      "package_id": "ShanghaiTech_03_0039",
      "dataset": "shanghaitech",
      "frames": 481,
      "labels": [
        "T05"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_03_0039"
    },
    {
      "package_id": "ShanghaiTech_07_0007",
      "dataset": "shanghaitech",
      "frames": 481,
      "labels": [
        "T01"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_07_0007"
    },
    {
      "package_id": "ShanghaiTech_01_0139",
      "dataset": "shanghaitech",
      "frames": 217,
      "labels": [
        "T02"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/shanghaitech/ShanghaiTech_01_0139"
    },
    {
      "package_id": "NWPU_D043_02",
      "dataset": "nwpu",
      "frames": 476,
      "labels": [
        "T03"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/nwpu/NWPU_D043_02"
    },
    {
      "package_id": "UCF-Crime_Assault_Assault024_x264",
      "dataset": "ucf_crime",
      "frames": 446,
      "labels": [
        "T04"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/ucf_crime/UCF-Crime_Assault_Assault024_x264"
    },
    {
      "package_id": "Avenue_11",
      "dataset": "avenue",
      "frames": 472,
      "labels": [
        "T05"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/avenue/Avenue_11"
    }
  ],
  "openset": [
    {
      "package_id": "NWPU_D003_05",
      "dataset": "nwpu",
      "frames": 2101,
      "labels": [
        "R06"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/nwpu/NWPU_D003_05"
    },
    {
      "package_id": "NWPU_D038_02",
      "dataset": "nwpu",
      "frames": 1201,
      "labels": [
        "R06"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/nwpu/NWPU_D038_02"
    },
    {
      "package_id": "NWPU_D013_01",
      "dataset": "nwpu",
      "frames": 2851,
      "labels": [
        "R06"
      ],
      "path": "/home/expand_disk/data_repository/mfl/token_compression/20260613_data/packages/nwpu/NWPU_D013_01"
    }
  ],
  "selection_params": {
    "train_count": 32,
    "val_count": 10,
    "openset_count": 3,
    "max_frames": 1600,
    "seed": 20260614
  }
}
```

## Sample Summary

```json
{
  "missing_packages": [],
  "package_splits": {
    "train": [
      "Avenue_21",
      "ShanghaiTech_06_0145",
      "ShanghaiTech_01_0132",
      "UCF-Crime_Assault_Assault039_x264",
      "ShanghaiTech_05_0017",
      "ShanghaiTech_01_0055",
      "ShanghaiTech_01_0162",
      "NWPU_D068_02",
      "ShanghaiTech_01_0052",
      "ShanghaiTech_09_0057",
      "Avenue_18",
      "ShanghaiTech_12_0173",
      "ShanghaiTech_01_0130",
      "ShanghaiTech_07_0048",
      "ShanghaiTech_07_0008",
      "ShanghaiTech_07_0009",
      "ShanghaiTech_01_0133",
      "ShanghaiTech_01_0016",
      "ShanghaiTech_07_0006",
      "Avenue_20",
      "Avenue_08",
      "ShanghaiTech_06_0153",
      "ShanghaiTech_12_0148",
      "NWPU_D094_01",
      "ShanghaiTech_03_0031",
      "ShanghaiTech_03_0041",
      "ShanghaiTech_01_0014",
      "NWPU_D047_06",
      "ShanghaiTech_03_0033",
      "ShanghaiTech_08_0079",
      "ShanghaiTech_03_0035",
      "ShanghaiTech_06_0144"
    ],
    "val": [
      "ShanghaiTech_01_0027",
      "ShanghaiTech_01_0163",
      "ShanghaiTech_06_0147",
      "ShanghaiTech_05_0024",
      "ShanghaiTech_03_0039",
      "ShanghaiTech_07_0007",
      "ShanghaiTech_01_0139",
      "NWPU_D043_02",
      "UCF-Crime_Assault_Assault024_x264",
      "Avenue_11"
    ],
    "openset": [
      "NWPU_D003_05",
      "NWPU_D038_02",
      "NWPU_D013_01"
    ]
  },
  "counts": {
    "split_train": 143,
    "label_T01": 12,
    "type_object_track": 179,
    "positive_True": 92,
    "label_normal": 87,
    "positive_False": 87,
    "label_T02": 19,
    "label_T03": 17,
    "label_T04": 31,
    "label_T05": 9,
    "split_val": 32,
    "split_openset": 4,
    "label_R06": 4
  }
}
```

## Metrics

```json
[
  {
    "ablation": "anomaly_vector_visual_only",
    "strategy": "anomaly_vectors_only_low_score_is_normal",
    "anomaly_vectors": 8,
    "best_row": {
      "epoch": 50,
      "loss": 0.12294454687320581,
      "bce": 0.1225049144790306,
      "sep": 0.004396336029407444,
      "val_balanced_accuracy": 0.96875,
      "val_anomaly_recall": 0.9375,
      "val_normal_fpr": 0.0,
      "val_auroc": 0.97265625,
      "val_event_top3_recall": 1.0
    },
    "train": {
      "n": 143,
      "threshold": 0.5,
      "accuracy": 0.965034965034965,
      "anomaly_recall": 0.9305555555555556,
      "normal_false_positive_rate": 0.0,
      "normal_true_negative_rate": 1.0,
      "auroc": 0.9980438184663537,
      "per_label": {
        "normal": {
          "count": 71,
          "mean_score": 0.012909564189612865,
          "pred_anomaly_rate": 0.0
        },
        "T01": {
          "count": 10,
          "mean_score": 0.9980367422103882,
          "pred_anomaly_rate": 1.0
        },
        "T02": {
          "count": 15,
          "mean_score": 0.9987090229988098,
          "pred_anomaly_rate": 1.0
        },
        "T03": {
          "count": 14,
          "mean_score": 0.9990015625953674,
          "pred_anomaly_rate": 1.0
        },
        "T04": {
          "count": 26,
          "mean_score": 0.7914273738861084,
          "pred_anomaly_rate": 0.8076923076923077
        },
        "T05": {
          "count": 7,
          "mean_score": 0.995536744594574,
          "pred_anomaly_rate": 1.0
        }
      },
      "event_top1_recall": 1.0,
      "event_top3_recall": 1.0,
      "event_ranking_count": 34
    },
    "val": {
      "n": 32,
      "threshold": 0.5,
      "accuracy": 0.96875,
      "anomaly_recall": 0.9375,
      "normal_false_positive_rate": 0.0,
      "normal_true_negative_rate": 1.0,
      "auroc": 0.97265625,
      "per_label": {
        "normal": {
          "count": 16,
          "mean_score": 0.055831942707300186,
          "pred_anomaly_rate": 0.0
        },
        "T01": {
          "count": 2,
          "mean_score": 0.9991745948791504,
          "pred_anomaly_rate": 1.0
        },
        "T02": {
          "count": 4,
          "mean_score": 0.9991585612297058,
          "pred_anomaly_rate": 1.0
        },
        "T03": {
          "count": 3,
          "mean_score": 0.9989818930625916,
          "pred_anomaly_rate": 1.0
        },
        "T04": {
          "count": 5,
          "mean_score": 0.7995734810829163,
          "pred_anomaly_rate": 0.8
        },
        "T05": {
          "count": 2,
          "mean_score": 0.9974707365036011,
          "pred_anomaly_rate": 1.0
        }
      },
      "event_top1_recall": 1.0,
      "event_top3_recall": 1.0,
      "event_ranking_count": 10
    },
    "openset": {
      "n": 4,
      "mean_anomaly_score": 0.2552870512008667,
      "max_anomaly_score": 0.99906986951828,
      "min_anomaly_score": 0.0042161764577031136
    },
    "num_train": 143,
    "num_val": 32,
    "num_openset": 4
  },
  {
    "ablation": "anomaly_vector_visual_motion",
    "strategy": "anomaly_vectors_only_low_score_is_normal",
    "anomaly_vectors": 8,
    "best_row": {
      "epoch": 3,
      "loss": 0.6077292886647311,
      "bce": 0.6077292886647311,
      "sep": 0.0,
      "val_balanced_accuracy": 0.9375,
      "val_anomaly_recall": 0.9375,
      "val_normal_fpr": 0.0625,
      "val_auroc": 0.953125,
      "val_event_top3_recall": 1.0
    },
    "train": {
      "n": 143,
      "threshold": 0.5,
      "accuracy": 0.8671328671328671,
      "anomaly_recall": 0.9027777777777778,
      "normal_false_positive_rate": 0.16901408450704225,
      "normal_true_negative_rate": 0.8309859154929577,
      "auroc": 0.9096244131455399,
      "per_label": {
        "normal": {
          "count": 71,
          "mean_score": 0.39097192883491516,
          "pred_anomaly_rate": 0.16901408450704225
        },
        "T01": {
          "count": 10,
          "mean_score": 0.5731691122055054,
          "pred_anomaly_rate": 0.8
        },
        "T02": {
          "count": 15,
          "mean_score": 0.6218616366386414,
          "pred_anomaly_rate": 0.8666666666666667
        },
        "T03": {
          "count": 14,
          "mean_score": 0.6072225570678711,
          "pred_anomaly_rate": 1.0
        },
        "T04": {
          "count": 26,
          "mean_score": 0.6084572076797485,
          "pred_anomaly_rate": 0.9230769230769231
        },
        "T05": {
          "count": 7,
          "mean_score": 0.5509325861930847,
          "pred_anomaly_rate": 0.8571428571428571
        }
      },
      "event_top1_recall": 0.9411764705882353,
      "event_top3_recall": 1.0,
      "event_ranking_count": 34
    },
    "val": {
      "n": 32,
      "threshold": 0.5,
      "accuracy": 0.9375,
      "anomaly_recall": 0.9375,
      "normal_false_positive_rate": 0.0625,
      "normal_true_negative_rate": 0.9375,
      "auroc": 0.953125,
      "per_label": {
        "normal": {
          "count": 16,
          "mean_score": 0.40250909328460693,
          "pred_anomaly_rate": 0.0625
        },
        "T01": {
          "count": 2,
          "mean_score": 0.6476549506187439,
          "pred_anomaly_rate": 1.0
        },
        "T02": {
          "count": 4,
          "mean_score": 0.645485520362854,
          "pred_anomaly_rate": 1.0
        },
        "T03": {
          "count": 3,
          "mean_score": 0.6386446952819824,
          "pred_anomaly_rate": 1.0
        },
        "T04": {
          "count": 5,
          "mean_score": 0.511414647102356,
          "pred_anomaly_rate": 0.8
        },
        "T05": {
          "count": 2,
          "mean_score": 0.5607452392578125,
          "pred_anomaly_rate": 1.0
        }
      },
      "event_top1_recall": 1.0,
      "event_top3_recall": 1.0,
      "event_ranking_count": 10
    },
    "openset": {
      "n": 4,
      "mean_anomaly_score": 0.4856148958206177,
      "max_anomaly_score": 0.6707462668418884,
      "min_anomaly_score": 0.305683970451355
    },
    "num_train": 143,
    "num_val": 32,
    "num_openset": 4
  },
  {
    "ablation": "anomaly_vector_token_topk",
    "strategy": "anomaly_vectors_only_low_score_is_normal",
    "anomaly_vectors": 8,
    "best_row": {
      "epoch": 66,
      "loss": 0.024779503098591327,
      "bce": 0.024560793335924015,
      "sep": 0.00218709803046333,
      "val_balanced_accuracy": 1.0,
      "val_anomaly_recall": 1.0,
      "val_normal_fpr": 0.0,
      "val_auroc": 1.0,
      "val_event_top3_recall": 1.0
    },
    "train": {
      "n": 143,
      "threshold": 0.5,
      "accuracy": 0.986013986013986,
      "anomaly_recall": 1.0,
      "normal_false_positive_rate": 0.028169014084507043,
      "normal_true_negative_rate": 0.971830985915493,
      "auroc": 0.9999021909233177,
      "per_label": {
        "normal": {
          "count": 71,
          "mean_score": 0.022044308483600616,
          "pred_anomaly_rate": 0.028169014084507043
        },
        "T01": {
          "count": 10,
          "mean_score": 0.9991782307624817,
          "pred_anomaly_rate": 1.0
        },
        "T02": {
          "count": 15,
          "mean_score": 0.9995530843734741,
          "pred_anomaly_rate": 1.0
        },
        "T03": {
          "count": 14,
          "mean_score": 0.9995504021644592,
          "pred_anomaly_rate": 1.0
        },
        "T04": {
          "count": 26,
          "mean_score": 0.9733608961105347,
          "pred_anomaly_rate": 1.0
        },
        "T05": {
          "count": 7,
          "mean_score": 0.9992865920066833,
          "pred_anomaly_rate": 1.0
        }
      },
      "event_top1_recall": 1.0,
      "event_top3_recall": 1.0,
      "event_ranking_count": 34
    },
    "val": {
      "n": 32,
      "threshold": 0.5,
      "accuracy": 1.0,
      "anomaly_recall": 1.0,
      "normal_false_positive_rate": 0.0,
      "normal_true_negative_rate": 1.0,
      "auroc": 1.0,
      "per_label": {
        "normal": {
          "count": 16,
          "mean_score": 0.020509425550699234,
          "pred_anomaly_rate": 0.0
        },
        "T01": {
          "count": 2,
          "mean_score": 0.999448299407959,
          "pred_anomaly_rate": 1.0
        },
        "T02": {
          "count": 4,
          "mean_score": 0.999589741230011,
          "pred_anomaly_rate": 1.0
        },
        "T03": {
          "count": 3,
          "mean_score": 0.9995281100273132,
          "pred_anomaly_rate": 1.0
        },
        "T04": {
          "count": 5,
          "mean_score": 0.9004623293876648,
          "pred_anomaly_rate": 1.0
        },
        "T05": {
          "count": 2,
          "mean_score": 0.9993072748184204,
          "pred_anomaly_rate": 1.0
        }
      },
      "event_top1_recall": 1.0,
      "event_top3_recall": 1.0,
      "event_ranking_count": 10
    },
    "openset": {
      "n": 4,
      "mean_anomaly_score": 0.25271075963974,
      "max_anomaly_score": 0.9990214109420776,
      "min_anomaly_score": 0.003364378120750189
    },
    "num_train": 143,
    "num_val": 32,
    "num_openset": 4
  }
]
```
