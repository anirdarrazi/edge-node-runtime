import os
from node_agent.vast_smoke import VastAPI
instance_id = int(os.environ["ENDURANCE_VAST_INSTANCE_ID"])
api = VastAPI(os.environ["VAST_API_KEY"])
api.destroy_instance(instance_id)
print(f"destroyed_instance={instance_id}")
