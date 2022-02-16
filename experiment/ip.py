import ifaddr
import os

if os.getenv("SLURM_PROCID") == 0:
    adapters = ifaddr.get_adapters()

    for adapter in adapters:
        if adapter.nice_name == "ipogif0":
            for addr in adapter.ips:
                if ':' not in addr.ip:
                    print(f"{addr.ip}")