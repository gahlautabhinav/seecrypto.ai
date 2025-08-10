# weekly_trainer.py

import os
from lstm_training import main
from datetime import datetime
import logging
import time

# ========== Configuration ==========
coin_list = [
    # 19 previous
    "bitcoin", "ethereum", "cardano", "ripple", "solana",
    "dogecoin", "litecoin", "chainlink", "polkadot", "uniswap",
    "avalanche-2", "stellar", "cosmos", "tron", "vechain",
    "the-graph", "aave", "optimism", "arbitrum",
    
    # 20 more
    "aptos", "sui", "algorand", "hedera-hashgraph", "near",
    "render-token", "mina-protocol", "fantom", "theta-token", "axie-infinity",
    "flow", "immutable-x", "lido-dao", "gala", "chiliz",
    "oasis-network", "iota", "zilliqa", "basic-attention-token", "enjincoin"
]

# ========== Logging ==========
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/weekly_train_{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# ========== Run Weekly Training ==========
if __name__ == "__main__":
    logging.info("🔁 Starting Weekly Model Retraining...\n")
    for coin in coin_list:
        try:
            logging.info(f"🚀 Retraining model for: {coin}")
            main(coin)
            logging.info(f"✅ Retraining completed for: {coin}\n")
            time.sleep(5)
        except Exception as e:
            logging.error(f"❌ Failed retraining for {coin}: {e}\n")
            time.sleep(15)
    logging.info("🎉 Weekly retraining complete!")
