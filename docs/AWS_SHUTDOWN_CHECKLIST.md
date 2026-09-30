# AWS shutdown checklist

What to do when the AWS credits are about to run out. Start **two weeks before** the credit end date
(the plan assumed February). Everything below is safe to do in order, and nothing else in the project
depends on AWS: the code and the public dashboard keep working afterwards.

Resources this project created (all in `ca-central-1`, account `683803166018`):

| Resource | Name / ID |
| --- | --- |
| EC2 server (k3s) | `i-0a73e5abe56cf13b3` |
| Security group | `sg-0f45fe0cf33ce6050` |
| S3 bucket | `oulad-mlops-jasleenminhas578-22230` |
| IAM role for the server + instance profile | `oulad-ec2-role`, `oulad-ec2-profile` |
| IAM role for GitHub deploys | `oulad-github-deploy` |
| Budget alert | `oulad-monthly-20usd` (if created) |

## 1. Find out the real deadline (do this first)

In the AWS console open **Billing and Cost Management**, then **Credits** (and **Free Plan** if shown).
Note the credit expiry date and what happens when the plan ends. On a free-plan account, resources can be
stopped or the account closed when credits or the plan run out, so do not wait for the last day.
Put the deadline in your calendar, with a reminder two weeks before.

## 2. Save what you want to keep (about 10 minutes)

Do this while the server still exists. Start it if it is stopped:

```bash
aws ec2 start-instances --region ca-central-1 --instance-ids i-0a73e5abe56cf13b3
```

Its public IP changes on every start. Get the new one:

```bash
aws ec2 describe-instances --region ca-central-1 --instance-ids i-0a73e5abe56cf13b3 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

Your own IP may also have changed, so open the ports again if the API or dashboard will not load
(replace the IP): `aws ec2 authorize-security-group-ingress --region ca-central-1 --group-id sg-0f45fe0cf33ce6050 --protocol tcp --port 30080 --cidr <YOUR_IP>/32` (and port `30501` for the dashboard).

Then download the results:

```bash
B=oulad-mlops-jasleenminhas578-22230
mkdir -p aws-backup
aws s3 sync s3://$B/data/state aws-backup/state
aws s3 sync s3://$B/data/reports aws-backup/reports
aws s3 sync s3://$B/data/predictions aws-backup/predictions
aws s3 sync s3://$B/models aws-backup/models
```

Also take screenshots for the README and demo (these cannot be recreated after teardown):

- The API docs page (`http://<server-ip>:30080/docs`) and `/health` showing the model version.
- The dashboard on the AWS server (`http://<server-ip>:30501`).
- The MLflow UI, opened through an SSM tunnel to port 30500, showing the registered model and its `champion` alias.
- The GitHub **Actions** tab showing a green `deploy` run.
- `kubectl -n oulad get pods` from a Session Manager shell on the server.

Save the screenshots in `docs/` and commit them.

## 3. Refresh the public dashboard (optional)

The public Streamlit dashboard reads `dashboard/snapshot/`. To update it from the final run, either copy
`aws-backup` files into the same layout (`data/state`, `data/processed`, `data/predictions`,
`data/reports`, `models/champion`, `models/v1`) or re-run the pipeline locally and run
`python scripts/make_snapshot.py`. Commit and push; Streamlit Cloud redeploys by itself.

## 4. Update the repository before tearing down

1. In `README.md`, change the AWS wording to say the AWS deployment was taken down after the credit
   window and that the public dashboard runs from a saved snapshot. Keep the screenshots.
2. Stop the deploy workflow from failing on every push: in `.github/workflows/deploy.yml` change the
   `on:` block to only `workflow_dispatch:`, or delete the file.
3. Delete the repository secrets `AWS_DEPLOY_ROLE_ARN`, `AWS_REGION` and `EC2_INSTANCE_ID`
   (GitHub repository, Settings, Secrets and variables, Actions).
4. Revoke the fine-grained GitHub token used for redeploy events
   (GitHub, Settings, Developer settings, Fine-grained tokens, Delete).
5. Commit and push with a plain commit message.

## 5. Tear down AWS

This deletes everything permanently. Make sure step 2 is done.

```bash
R=ca-central-1
aws ec2 terminate-instances --region $R --instance-ids i-0a73e5abe56cf13b3
aws ec2 wait instance-terminated --region $R --instance-ids i-0a73e5abe56cf13b3
aws ec2 delete-security-group --region $R --group-id sg-0f45fe0cf33ce6050
aws s3 rb s3://oulad-mlops-jasleenminhas578-22230 --force
aws iam remove-role-from-instance-profile --instance-profile-name oulad-ec2-profile --role-name oulad-ec2-role
aws iam delete-instance-profile --instance-profile-name oulad-ec2-profile
aws iam delete-role-policy --role-name oulad-ec2-role --policy-name oulad-s3
aws iam detach-role-policy --role-name oulad-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam delete-role --role-name oulad-ec2-role
aws iam delete-role-policy --role-name oulad-github-deploy --policy-name oulad-ssm
aws iam delete-role --role-name oulad-github-deploy
aws budgets delete-budget --account-id 683803166018 --budget-name oulad-monthly-20usd   # optional, once billing is zero
```

The GitHub OIDC provider (`token.actions.githubusercontent.com`) already existed in the account before
this project, so it is not deleted here. Delete it only if nothing else uses it.

## 6. Check nothing is left running

```bash
R=ca-central-1
aws ec2 describe-instances --region $R --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text
aws ec2 describe-volumes --region $R --query 'Volumes[].[VolumeId,State,Size]' --output text
aws ec2 describe-addresses --region $R --query 'Addresses[].PublicIp' --output text
aws s3 ls
```

You want no running instances, no leftover volumes, no unattached Elastic IPs and no bucket from this
project. Check **Billing, Bills** again the next day: the current month should stop growing.

## What keeps working afterwards

- **GitHub repository**, README, code, tests and CI (lint, tests, image builds) at no cost.
- **Public dashboard** on Streamlit Community Cloud, running from `dashboard/snapshot/`.
- **Container images** on GHCR (free for a public repository).
- **The kind demo** on a laptop (`make kind-up kind-deploy`), which needs no cloud account.

What stops: the live API and MLflow on AWS, the scheduled pipeline, and automatic redeploys.

## If you want to keep it running after the credits

Leaving the server on costs roughly a few US cents per hour plus the public IP and disk (check the
EC2 pricing page for `ca-central-1`). Stopping it when idle brings this down to the cost of the 30 GB disk.
Otherwise, follow steps 4 to 6.
