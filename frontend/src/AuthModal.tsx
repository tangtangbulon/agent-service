import { LockOutlined, UserOutlined } from '@ant-design/icons';
import { Alert, Form, Input, Modal } from 'antd';
import { useState } from 'react';

import { login } from './api';
import type { AuthSession } from './types';

interface AuthModalProps {
  open: boolean;
  onCancel: () => void;
  onSuccess: (session: AuthSession) => void;
}

interface AuthFormValues {
  username: string;
  password: string;
}

export function AuthModal({
  open,
  onCancel,
  onSuccess,
}: AuthModalProps) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(values: AuthFormValues) {
    setSubmitting(true);
    setError(null);
    try {
      onSuccess(await login(values.username.trim(), values.password));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '登录失败，请稍后重试');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Modal
      open={open}
      title={null}
      footer={null}
      width={420}
      onCancel={onCancel}
      destroyOnHidden
      centered
    >
      <div className="auth-modal__header">
        <h2>欢迎来到安心保</h2>
        <p>登录后可使用智能顾问服务</p>
      </div>
      {error ? <Alert showIcon type="error" message={error} style={{ marginBottom: 16 }} /> : null}
      <Form<AuthFormValues>
        layout="vertical"
        onFinish={(values) => void submit(values)}
        requiredMark={false}
      >
        <Form.Item
          label="用户名"
          name="username"
          rules={[{ required: true, whitespace: true, message: '请输入用户名' }]}
        >
          <Input prefix={<UserOutlined />} size="large" autoComplete="username" />
        </Form.Item>
        <Form.Item
          label="密码"
          name="password"
          rules={[{ required: true, message: '请输入密码' }]}
        >
          <Input.Password prefix={<LockOutlined />} size="large" autoComplete="current-password" />
        </Form.Item>
        <button className="auth-submit" type="submit" disabled={submitting}>
          {submitting ? '登录中…' : '登录'}
        </button>
      </Form>
    </Modal>
  );
}
