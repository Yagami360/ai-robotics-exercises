# Isaac Sim のシミュレーター上に LeRobot の SO-ARMS ロボットを配置する

- URDF [Unified Robot Description Format] ファイル
    ロボットの物理的構造を記述するためのXMLベースのファイル形式です。

- USD [Universal Scene Description] ファイル
    Pixarが開発した3Dシーンを記述するための汎用的なオープンソースフォーマット。Isaac SIm では USD ファイルをサポートしている

## 方法

### Isaac Sim のシミュレーター GUI 上で行う場合

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

1. Isaac Sim の 空シミュレーターを起動する

    ```bash
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

### Isaac Sim のシミュレーター GUI 上で行う場合

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

1. SO101-ARMS ロボットのURDFファイルをUSDファイルに変換する<br>

1. SO101-ARMSを配置した IsaacSim のシーンを作成する

1. シミュレーターを起動する
