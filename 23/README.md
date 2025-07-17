# NVIDIA Cosmos の概要

<img width="1000" height="448" alt="Image" src="https://github.com/user-attachments/assets/7e73bbc7-14af-4600-839d-da2438b562bd" />

- 世界基盤モデル [WFM: World Foundation Model] を OSS で提供し、世界基盤モデルを利用した各種機能（合成データ生成・データ拡張・品質評価）を提供している
    - NVIDIA COSMOS における 世界基盤モデルとは？
        - フォトリアリスティックな3Dレンダリングでの動画生成を行うモデル。単に動画生成をおこなだけではなく、現実世界に近い物理法則（重力・衝突判定の影響など）を考慮した動画生成を行う
        - 以下の２種類のモデルがある（後述の Cusmos コンポーネント毎に異なるモデルを利用する）

            - 拡散モデルベースの世界基盤モデル
                <img width="625" height="274" alt="Image" src="https://github.com/user-attachments/assets/bfbf50b0-48e8-4781-adc2-65155ac0fb71" /><br>
                - フォトリアリスティックで現実世界に近い動画生成のためのモデル

            - 自己回帰モデル
                <img width="625" height="239" alt="Image" src="https://github.com/user-attachments/assets/d604c064-6da3-4740-8b14-69bc5d893b72" /><br>
                
                - 入力テキストと過去の動画フレームに基づいて次の行動やフレームを予測する
            - また「Nano」、「Super」、「Ultra」のモデルサイズが提供されている

        - 活用用途
            - 現実世界に近い合成データセット生成
                - 後述の Cosmos Predict

            - データ拡張
                - 後述の Cosmos Transfer
                - xxx

- COSMOS のコンポーネント<br>

    - Cosmos Predict-1<br>

        https://github.com/nvidia-cosmos/cosmos-predict1

        Cosmos の主要コンポーネント。世界基盤モデルを用いて、以下の（フォトリアリスティックで物理的に正確な）動画生成タスクを行え、合成データ生成に活用できる

        - text-to-world<br>

            https://github.com/user-attachments/assets/aa4b391a-234a-4d97-9284-b48648bfa84a

            - テキストを入力として、（フォトリアリスティックで物理的に正確な）動画を生成するタスク

        - video-to-world<br>

            https://github.com/user-attachments/assets/4c71797b-04d9-4a91-adfc-a5ecaabf9000

            - 動画を入力として、（フォトリアリスティックで物理的に正確な）未来の行動やフレームを生成するタスク

    - Cosmos Predict-2<br>
        https://github.com/nvidia-cosmos/cosmos-predict2

        Cosmos Predict-1 の後発コンポーネントで、処理パフォーマンスを大幅改善したコンポーネント

    - Cosmos Transfer<br>

        https://github.com/nvidia-cosmos/cosmos-transfer1
        
        [Cosmos-Transfer1 - a nvidia Collection](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e)

        世界基盤モデルを用いて、様々なフォトリアリスティックな動画生成を行なうことでデータ拡張（Data augument）を行なうコンポーネント

        https://github.com/user-attachments/assets/8c471204-e459-4f71-a65a-7459aaac1c05

    - Cosmos Reason<br>
        合成データの品質評価を行うコンポーネント
