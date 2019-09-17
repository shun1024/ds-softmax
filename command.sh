# 100 (10 x 10) Classes Toy Example
python main.py --learning_rate=0.003 --epoch=20 --nexperts=100 --nsuperclass=100 --lasso_loss_coef=0.002 --expert_lasso_loss_coef=0.002 --importance_loss_coef=10 --pruning_start=3 --pruning_cutoff=0.03

# 10000 (100 x 100) Classes Toy Example
python main.py --learning_rate=0.003 --nexperts=10 --nsuperclass=10 --lasso_loss_coef=0.005 --expert_lasso_loss_coef=0.005 --importance_loss_coef=10 --pruning_start=3 --pruning_cutoff=0.03

