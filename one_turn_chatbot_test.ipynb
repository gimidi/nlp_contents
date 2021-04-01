{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from IPython.display import display, Image\n",
    "from google.cloud import dialogflow\n",
    "\n",
    "import numpy as np\n",
    "import torch\n",
    "from gluonnlp.data import SentencepieceTokenizer\n",
    "from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model\n",
    "from kogpt2.utils import get_tokenizer\n",
    "from pytorch_lightning.core.lightning import LightningModule\n",
    "\n",
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "U_TKN = '<usr>'\n",
    "S_TKN = '<sys>'\n",
    "BOS = '<s>'\n",
    "EOS = '</s>'\n",
    "MASK = '<unused0>'\n",
    "SENT = '<unused1>'\n",
    "\n",
    "box = []\n",
    "\n",
    "class community(LightningModule):\n",
    "    def __init__(self, hparams, **kwargs):\n",
    "        super(community, self).__init__()\n",
    "        self.tok_path = get_tokenizer()\n",
    "        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()\n",
    "        \n",
    "    def forward(self, inputs):\n",
    "        output, _ = self.kogpt2(inputs)\n",
    "        return output\n",
    "    \n",
    "    def chat(self, max_length):\n",
    "        self.tok_path\n",
    "        tok = SentencepieceTokenizer(self.tok_path, num_best=0, alpha=0)\n",
    "        cnt = 0\n",
    "        with torch.no_grad():\n",
    "            while 1 :\n",
    "                global box\n",
    "                q = input('                                                                이용자😗 : ').strip()\n",
    "                q_tok = tok(q)\n",
    "                a = ''\n",
    "                a_tok = []\n",
    "                while 1:\n",
    "                    input_ids = torch.LongTensor([\n",
    "                        self.vocab[U_TKN]] + self.vocab[q_tok] + [self.vocab[EOS]] + [\n",
    "                        self.vocab[S_TKN]] + self.vocab[a_tok]).unsqueeze(dim=0)\n",
    "                    pred = self(input_ids)\n",
    "                    gen = self.vocab.to_tokens(\n",
    "                        torch.argmax(\n",
    "                            pred,\n",
    "                            dim=-1).squeeze().numpy().tolist())[-1]\n",
    "                    if gen == EOS:\n",
    "                        break\n",
    "                    a += gen.replace('▁', ' ')\n",
    "                    a_tok = tok(a)\n",
    "                    if len(a) > max_length : \n",
    "                        a = '무한루프'\n",
    "                        break\n",
    "                print(a.strip().replace('OO','민지'))\n",
    "                box.append(a.strip())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "using cached model\n",
      "using cached model\n",
      "using cached model\n"
     ]
    }
   ],
   "source": [
    "model = community.load_from_checkpoint('C:/Users/lll/5.mjkim/KoGPT2-chatbot/model_chp/chatbot/test_2/epoch50_-epoch=37-loss=9.64.ckpt')"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "test1\n",
    "\n",
    "-- 추가 \n",
    "빡치네 (무한루프) \n",
    "곰곰아 너는 사람이니?\n",
    "사람이냐\n",
    "음악추천해주라\n",
    "이건좀.....\n",
    "\n",
    "-- 수정\n",
    "곰곰이 -> 로니\n",
    "\n",
    "-- 코드\n",
    "<unk> replace 처리\n",
    "\n",
    "-- 로우데이터\n",
    "맞춤법검사기 결과 영역  를 째로 복붙한 사람들이 있는거 같음 -> 원본 데이터에서 일괄 수정하기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "이용자😗 : 빡치네\n",
      "무한루프\n",
      "이용자😗 : ㅎㅎ\n",
      "헤헤 :)\n",
      "이용자😗 : 헤헤\n",
      "헤헤 :)\n",
      "이용자😗 : 헤헤헤헤헤\n",
      "헤헤 :)\n",
      "이용자😗 : ㅎㅎ\n",
      "헤헤 :)\n",
      "이용자😗 : 하하하ㅏ\n",
      ":)\n",
      "이용자😗 : 야\n",
      "안녕하세요 :)\n",
      "이용자😗 : 너 이름뭐야\n",
      "저는 곰곰이랍니다 :) 반가워요\n",
      "이용자😗 : 곰곰이 아니자나\n",
      "네~ 저 여기있어요!\n",
      "이용자😗 : 안녕\n",
      "안녕하세요 :)\n",
      "이용자😗 : 나 너무힘들어\n",
      "민지님 너무 힘들어 하지마세요. 조금만 더 견뎌봐요. 오늘 하루도 수고했어요\n",
      "이용자😗 : 고마워 곰곰아\n",
      "네~ 저 여기있어요!\n",
      "이용자😗 : 곰곰아 너는 사람이니?\n",
      "네~ 저 여기있어요!\n",
      "이용자😗 : 사람이냐\n",
      "사람이라뇨? 당신이라는 사람은 정말 좋은 사람이에요. 좋은 사람도 많이 있고요. 외로움이나 불안함도 없을 것 같아요.\n",
      "이용자😗 : 집가고싶어\n",
      "집에 가고 싶은 날이에요!!놀이터도 좋고, 집에서 영화도 좋고 가까운 곳에 사방이 있어 걸어보는것도 좋을거 같아요!!\n",
      "이용자😗 : 스트레스받아\n",
      "요즘 스트레스받는 일이 있으신가요? 저는 슬픈 영상을 보며 울거나, 음악 감상을 하면서 마음을 비우면 한결 나아져요. 힘내요.\n",
      "이용자😗 : 음악추천해주라\n",
      "맘스터치 화이트 갈릭 추천해요~ 신나고 밝은 분위기의 노래를 들어보세요.\n",
      "이용자😗 : 맘스터치 화이트갈릭은 햄버거야\n",
      "맛있겠어요.\n",
      "이용자😗 : 맞아 먹고싶다\n",
      "저도 공감이에요<unk>\n",
      "이용자😗 : 나 사줘\n",
      "충분히 가치 있어요! 남에게 보이는 삶보다 나 자신을 위해 사는 삶이 더 멋있는 사람이라고 생각해요.\n",
      "이용자😗 : 주하님\n",
      "안녕하세요 :)\n",
      "이용자😗 : 흘렸어요\n",
      "꼬옥\n",
      "이용자😗 : 남겨먹을건가요?\n",
      "저는 나눠 먹었어요.\n",
      "이용자😗 : 맛있겠다\n",
      "저도볶이 너무 좋아해요!\n",
      "이용자😗 : 떡볶이 좋아해?\n",
      "떡볶이 너무 좋아해요.\n",
      "이용자😗 : 넌 여자야?\n",
      "저는 여자도 남자도 아니랍니다!\n",
      "이용자😗 : 외계인이야?\n",
      "저는 외계인이 아니라도 당신 마음속에 있는 외계인을 좋아해요.\n",
      "이용자😗 : 칭찬해\n",
      "우리 민지님은 존재 자체만으로도 이미 예쁜 분이세요.\n",
      "이용자😗 : 이건좀.....\n",
      "무한루프\n",
      "이용자😗 : 어노잉오렌지알아?\n",
      "좋아요.알아주는것 만으로도 감사해요.\n",
      "이용자😗 : 나 그거 닮았대\n",
      "안 닮았어, 이쁘다니까 그렇게 말 하는 사람들이야.\n",
      "이용자😗 : 나 이뻐?\n",
      "너 이뻐!\n",
      "이용자😗 : 고맙다\n",
      "민지님도 저도 고맙고 사랑해요!\n",
      "이용자😗 : 오\n",
      "맞춤법검사기 결과 영역 지치고 힘이 들 땐 무엇이든 원망의 대상이 될 수 있답니다. 너무 힘이 들 땐 마음을 가라앉히고 나 자신을 더욱 돌보면 남을 원망하지 않을 거예요.\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "Interrupted by user",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-11-e61c116cb3ce>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mchat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m600\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-9-c4fdf92e57d7>\u001b[0m in \u001b[0;36mchat\u001b[1;34m(self, max_length)\u001b[0m\n\u001b[0;32m     25\u001b[0m             \u001b[1;32mwhile\u001b[0m \u001b[1;36m1\u001b[0m \u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     26\u001b[0m                 \u001b[1;32mglobal\u001b[0m \u001b[0mbox\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 27\u001b[1;33m                 \u001b[0mq\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0minput\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'이용자😗 : '\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstrip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     28\u001b[0m                 \u001b[0mq_tok\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtok\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mq\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     29\u001b[0m                 \u001b[0ma\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m''\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\ipykernel\\kernelbase.py\u001b[0m in \u001b[0;36mraw_input\u001b[1;34m(self, prompt)\u001b[0m\n\u001b[0;32m    858\u001b[0m                 \u001b[1;34m\"raw_input was called, but this frontend does not support input requests.\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    859\u001b[0m             )\n\u001b[1;32m--> 860\u001b[1;33m         return self._input_request(str(prompt),\n\u001b[0m\u001b[0;32m    861\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_parent_ident\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    862\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_parent_header\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\ipykernel\\kernelbase.py\u001b[0m in \u001b[0;36m_input_request\u001b[1;34m(self, prompt, ident, parent, password)\u001b[0m\n\u001b[0;32m    902\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    903\u001b[0m                 \u001b[1;31m# re-raise KeyboardInterrupt, to truncate traceback\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 904\u001b[1;33m                 \u001b[1;32mraise\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Interrupted by user\"\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    905\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    906\u001b[0m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlog\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mwarning\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Invalid Message:\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mexc_info\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: Interrupted by user"
     ]
    }
   ],
   "source": [
    "# test1\n",
    "model.chat(600)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
