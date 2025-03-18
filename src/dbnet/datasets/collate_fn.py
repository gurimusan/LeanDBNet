import torch


def collate_fn(samples: list[tuple[torch.Tensor, any]]) -> tuple[torch.Tensor, list[any]]:
    images, targets = zip(*samples)

    max_h = 0
    max_w = 0
    for img in images:
        h, w = img.shape[1:]
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    # バックボーンネットワークで特徴マップの解像度を下げるときに
    # 切り捨てが起きないように入力の幅と高さを32の倍数にしておく
    # もし32の倍数でない場合、バックボーンネットワークの特徴マップと
    # 特徴ピラミッドネットワークのアップスケーリングでできた特徴マップ
    # の大きさに不整合が生じ、加算できなくなる
    max_h = (max_h + 31) // 32 * 32
    max_w = (max_w + 31) // 32 * 32

    # 画像を一つにテンソルにまとめ、ラベルはリストに集約
    imgs = images[0].new_zeros((len(images), 3, max_h, max_w))
    for i, img in enumerate(images):
        h, w = img.shape[1:]
        imgs[i, :, :h, :w] = img

    return imgs, list(targets)
