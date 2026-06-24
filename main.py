from r2ai.cli import main


if __name__ == "__main__":
    main()
# from qdrant_client import QdrantClient
# from qdrant_client.models import HnswConfigDiff

# client = QdrantClient(
#     url="https://71b57332-7eb8-4741-a8e6-44f607c11a5c.us-west-1-0.aws.cloud.qdrant.io",
#     api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MDIwNWEyMGEtOTE3Yi00Zjk3LWI3MTEtNmVjNWFmOTUwOTM3In0.Y-mIU0RF7qKKgHpaacr1C5D7r4v6NVLAZ3C1Ctd3UaY"
# )

# client.update_collection(
#     collection_name="vld_business_law",
#     hnsw_config=HnswConfigDiff(
#         m=48
#     )
# )