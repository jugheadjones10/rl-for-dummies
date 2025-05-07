terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.27"
    }
  }

  required_version = ">= 0.14.9"
}

provider "aws" {
  profile = "default"
  region  = "ap-southeast-1"
}

module "rl_for_dummies" {
  source              = "./modules/rl_for_dummies"
  spot_bid_percentage = "50"
  instance_types = [
    # "g4dn.4xlarge", # 16 vCPU, 64GB, $1.204, GPU
    # "g4dn.xlarge",  # 4 vCPU, 16GB, $0.526, GPU
    # "r5ad.large",   # 2 vCPU, 16GB, $0.131
    # "c6a.4xlarge",  # 16 vCPU, 32GB
    # "c5a.large",    # 2 vCPU, 4GB, $0.077

    # ARM-based
    # "g5g.4xlarge",
    # "c6g.4xlarge",
    "c6g.16xlarge",
    "c6g.4xlarge",
    "g5g.2xlarge",
    # "c6g.medium",  # 1 vCPU, 2GB, $0.034
    # "m6gd.medium", # 1 vCPU, 4GB, $0.0452
  ]
}