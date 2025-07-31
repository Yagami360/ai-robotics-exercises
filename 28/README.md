# Isaac Sim のシミュレーター上に LeRobot の SO-ARMS ロボットを配置する

<!--
- URDF [Unified Robot Description Format] ファイル
    ロボットの物理的構造を記述するためのXMLベースのファイル形式です。

- USD [Universal Scene Description] ファイル
    Pixarが開発した3Dシーンを記述するための汎用的なオープンソースフォーマット。Isaac SIm では USD ファイルをサポートしている
-->

## 方法

### 事前準備

1. SO-ARM のレポジトリを clone する<br>

    ```bash
    git clone https://github.com/TheRobotStudio/SO-ARM100
    ```

    上記レポジトリの `SO-ARM100/Simulation/SO101` ディレクトリ以下に SO101-ARMS ロボットの各種 3Dオブジェクト関連のファイルが保存されているので、これを利用する

    ```bash
    + SO-ARM100/
      + Simulation/
        + SO100/
          + assets/                     # SO100-ARMS の3Dオブジェクトファイル
          | + --- *.part
          | + --- *.stl
          + --- scene.xml               # MuJoCo 用のシーンの XML ファイル？
          + --- so100_new_calib.xml     # MuJoCo 用のSO100-ARMS ロボットの XML ファイル？
          + --- so100_new_calib.urdf    # SO100-ARMS ロボットの URDF ファイル
        + SO101/
          + assets/                     # SO101-ARMS の3Dオブジェクトファイル
          | + --- *.part
          | + --- *.stl
          + --- scene.xml               # MuJoCo 用のシーンの XML ファイル？
          + --- so101_new_calib.xml     # MuJoCo 用のSO101-ARMS ロボットの XML ファイル？
          + --- so101_new_calib.urdf    # SO101-ARMS ロボットの URDF ファイル
    ```

### Isaac Sim のシミュレーター GUI 上で行う場合

#### URDF -> USD ファイルに変換して配置する場合

1. Isaac Sim の 空シミュレーターを起動する

    ```bash
    # VNC サーバーを使用する場合
    export DISPLAY=:1

    cd IsaacLab
    python scripts/tutorials/00_sim/create_empty.py
    ```

1. シミュレーター上にライトを配置する

    真っ暗で何も見えない環境のため、シミュレーターGUI上の「Create」->「Lights」->「Dome Light」からライトを配置する

    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/193838db-c957-4aa1-b913-da96b290bb1a" /><br>

1. SO101-ARMS ロボットの URDF ファイルを選択する<br>

    Isaac Sim のシミュレーター GUI 上の「Contents」->「My Computer」から、上記 clone したレポジトリの `so101_new_calib.urdf` ファイルを選択する

    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/f195689f-537a-43c9-827b-d7c8e5282d99" /><br>
    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/283104ef-2837-4dce-a452-0653c3573ac5" /><br>

1. SO101-ARMS ロボットのURDFファイルをUSDファイルに変換する<br>

    上記選択したURDFファイル（`so101_new_calib.urdf`）上で右クリックし、「Convert USD」ボタンを押してUSDファイルを変換して作成する

    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/fd70f135-206c-4591-a464-f512861ef56d" />


1. SO101-ARMS ロボットのUSDファイルをシミュレーター上に配置する

    変換後のUSDファイルは、`so101_new_calib/so101_new_calib.usd` ディレクトリ以下に保存されるので、この USD ファイルを選択後「Add Current Selection」をクリックして、シミュレーター上に配置する

    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/44c8b05f-b800-4b59-906e-5ab3e4dbc85b" />

    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/7a977062-c715-4415-9620-5bf3ba24610c" />


#### URDF ファイルで直接配置する場合

1. Isaac Sim の 空シミュレーターを起動する

    ```bash
    # VNC サーバーを使用する場合
    export DISPLAY=:1

    python IsaacLab/scripts/tutorials/00_sim/create_empty.py
    ```

1. シミュレーター上にライトを配置する

    真っ暗で何も見えない環境のため、シミュレーターGUI上の「Create」->「Lights」->「Dome Light」からライトを配置する

    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/193838db-c957-4aa1-b913-da96b290bb1a" /><br>

1. メニューバーーの「Window」->「Extentions」から拡張機能のページを開く

    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/cfed2dcd-1bee-4791-bf80-11014941bc4a" />


1. "urdf" 等のワードで検索し、「URDF IMPORTER EXTENTION」を有効化する

    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/c8c665a0-0826-4f8f-bc51-2defba87b356" /><br>
    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/2349786c-eb9d-4831-8bd1-b78224dd81e3" /><br>

1. メニューバーの「Import」から SO101-ARMS ロボットの URDF ファイルを選択して import する<br>

    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/c32235db-a10b-4c3e-ab3a-31fcdc59cadb" /><br>
    <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/89895dc2-8386-4e5e-beac-86f458c529d4" /><br>

    > 内部的には、URDF -> USD ファイルへの変換を行って import している模様


### Isaac Lab の Python コードで行なう場合

1. SO101-ARMS ロボットのURDFファイルをUSDファイルに変換する<br>

    - シミュレーター上 GUI 上で変換する場合

        1. 上記「Isaac Sim のシミュレーター GUI 上で行う場合」記載の方法で URDF -> USD ファイルに変換し、シミュレーター上に配置する

        1. 配置した USD の `root_joint` の `Articulation Root` を選択し、`☓` ボタンをクリックして削除する。その後「Add」->「Physics」->「Articulation Root」から再度作成する

            この操作は、後述の Isaac Lab を使用したシーン配置時に、`AttributeError: 'Articulation' object has no attribute 'has_external_wrench'` のエラーが発生するので、行なう必要あり

            ```bash
            - Issac Lab の API互換性の修正
                - 古いArticulation Root: Isaac Sim GUI変換時に作成された古いAPIスキーマ
                - 新しいArticulation Root: 現在のIsaac Lab/Isaac Simバージョンと互換性のあるAPIスキーマ
            ```

            <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/213c9207-429f-44ed-96a1-405d560febac" /><br>
            <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/833e049c-7776-45fd-b374-7cef4e64613c" /><br>

        1. シミュレーター上から `Articulation Root` を再設定した USD を選択し、「Save Selected」から Export する

            <img width="800" height="743" alt="Image" src="https://github.com/user-attachments/assets/f1583fd3-d786-4935-bdbe-891c296f2aec" /><br>
            <img width="1242" height="743" alt="Image" src="https://github.com/user-attachments/assets/4ca8b361-a7fb-42ab-a11f-0b9de82c0e05" /><br>

    - Isaac Lab 変換スクリプトを使用する場合

        > この方法だと変換後の USD ファイルで AttributeError: 'Articulation' object has no attribute 'has_external_wrench' のエラーが発生する

        1. Isaac Lab変換スクリプトにおいて、サポートしていないURDFファイル属性を除外する

            URDF ファイル内に、後段の変換スクリプトがサポートしていない以下のタグがあればそれらを削除する

            - `<gazebo>` タグ
            - `<transmission>` タグ

        1. URDFファイルをUSDファイルに変換するための Isaac Lab スクリプトを実行する<br>

            Isaac Lab が提供しているスクリプトを使用して、以下のコマンドで変換可能

            - パターン１
                ```bash
                cd IsaacLab
                ./isaaclab.sh -p scripts/tools/convert_urdf.py \
                    ../assets/so101_urdf/so101_new_calib_isaaclab.urdf \
                    ../assets/so101_usd/so101_new_calib.usd \
                    --merge-joints \
                    --joint-stiffness 0.0 \
                    --joint-damping 0.0 \
                    --joint-target-type none \
                    --headless
                ```

            - パターン２
                ```bash
                cd IsaacLab
                ./isaaclab.sh -p scripts/tools/convert_urdf.py \
                    ../assets/so101_urdf/so101_new_calib_isaaclab.urdf \
                    ../assets/so101_usd/so101_new_calib.usd \
                    --merge-joints \
                    --joint-stiffness 100.0 \
                    --joint-damping 10.0 \
                    --joint-target-type position \
                    --fix-base \
                    --headless
                ```

            - 参考情報：https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html#using-urdf-importer


1. SO101-ARMSを配置した IsaacSim のシーンのスクリプトを作成する

    [`scene_so101.py`](scene_so101.py)

1. シミュレーターを起動する

    ```python
    # VNCサーバーを使用する場合
    export DISPLAY=:1

    python scene_so101.py
    ```

    シミュレーション起動後に、以下のような SO101-ARMS が配置され、かつ各関節を動かしているシーンが表示される

    https://github.com/user-attachments/assets/b0c72c4a-adae-4462-b717-8bf7e3bde847

## 参考

- https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_urdf.html
- https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html#using-urdf-importer
- https://lycheeai-hub.com/project-so-arm101-x-isaac-sim-x-isaac-lab-tutorial-series/so-arm-import-urdf-to-isaac-sim
