# Model info — 3 backbone pre-trained ImageNet

> Tổng hợp từ `20261026-115633` (resnet18), `20261026-120218` (vgg11),
> `20261026-121817` (densenet121). Cấu hình chung: `strategy=freeze_backbone`,
> `freeze_until=1`, `freeze_head_full=false`, `num_classes=10`, `weights=DEFAULT`.

| Model | Tổng tham số | Train khi freeze | Tỉ lệ train | Output shape |
|-------|-------------:|-----------------:|------------:|--------------|
| ResNet18    | 11.181.642  | 5.130  | 0,046 % | [1, 10] |
| VGG11       | 128.807.306 | 40.970 | 0,032 % | [1, 10] |
| DenseNet121 | 6.964.106   | 10.250 | 0,147 % | [1, 10] |