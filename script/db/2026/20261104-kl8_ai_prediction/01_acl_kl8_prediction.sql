-- =============================================================================
-- 快乐8 (Keno) 深度学习与图谱贝叶斯混合集成预测系统 ACL 权限伴随脚本
-- 需求/模块: 20261104-kl8_ai_prediction
-- 作者: czw
-- 日期: 2026-11-04
-- =============================================================================

INSERT INTO `acl`.`acl_permission` (
    `APP_CODE`,
    `PERMISSION_KEY`,
    `PERMISSION_KEY_MD5`,
    `PERMISSION_NAME`,
    `PARENT_KEY`,
    `PERMISSION_TYPE`,
    `PATH`,
    `HTTP_METHOD`,
    `STATUS`,
    `MEMO`,
    `CREATOR`,
    `MODIFIER`,
    `GMT_CREATE`,
    `GMT_MODIFIED`
) VALUES
(
    'kl8_ai',
    'kl8:prediction:execute',
    MD5('kl8:prediction:execute'),
    '快乐8执行深度学习与图谱贝叶斯混合预测',
    'kl8:prediction',
    'BUTTON',
    '/api/v1/kl8/prediction/execute',
    'POST',
    1,
    '触发端到端 PyTorch Transformer 训练推理与 30 码级联缩水流水线',
    'czw',
    'czw',
    NOW(),
    NOW()
),
(
    'kl8_ai',
    'kl8:prediction:view',
    MD5('kl8:prediction:view'),
    '快乐8查看预测精要与三维交付物',
    'kl8:prediction',
    'MENU',
    '/api/v1/kl8/prediction/reports',
    'GET',
    1,
    '查询历史预测复盘、微信分桶文案、商业 Excel 报表与集群三元组',
    'czw',
    'czw',
    NOW(),
    NOW()
),
(
    'kl8_ai',
    'kl8:prediction:sync',
    MD5('kl8:prediction:sync'),
    '快乐8实时增量同步官方开奖数据',
    'kl8:prediction',
    'BUTTON',
    '/api/v1/kl8/history/sync',
    'POST',
    1,
    '从福彩官网或官方数据源实时增量获取最新一期开奖号码并更新历史库',
    'czw',
    'czw',
    NOW(),
    NOW()
)
ON DUPLICATE KEY UPDATE
    `PERMISSION_NAME` = VALUES(`PERMISSION_NAME`),
    `MEMO` = VALUES(`MEMO`),
    `MODIFIER` = VALUES(`MODIFIER`),
    `GMT_MODIFIED` = NOW();
