import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MarkdownMessage } from './MarkdownMessage';

describe('MarkdownMessage', () => {
  it('renders CommonMark and GFM structures without a separate HTML path', () => {
    const { container } = render(
      <MarkdownMessage
        content={`# 标题

- 列表一
- 列表二

> 引用内容

| 项目 | 状态 |
| --- | --- |
| 时间 | 完成 |

行内 \`code\`

\`\`\`ts
const value = 1
\`\`\`

[安全链接](https://example.com/path)`}
      />,
    );

    expect(screen.getByRole('heading', { name: '标题' })).toBeInTheDocument();
    expect(screen.getByRole('list')).toBeInTheDocument();
    expect(screen.getByText('引用内容').closest('blockquote')).not.toBeNull();
    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(container.querySelector('pre code')).toHaveTextContent('const value = 1');
    expect(screen.getByRole('link', { name: '安全链接' })).toHaveAttribute(
      'href',
      'https://example.com/path',
    );
    expect(screen.getByRole('link', { name: '安全链接' })).toHaveAttribute(
      'rel',
      'noreferrer noopener',
    );
  });

  it('does not execute raw HTML, dangerous URLs or remote markdown images', () => {
    const { container } = render(
      <MarkdownMessage
        content={`<script>window.__unsafe = true</script>

<strong>原始 HTML</strong>

[脚本链接](javascript:alert(1))

[数据链接](data:text/html,unsafe)

![远程像素](https://example.com/pixel.gif)`}
      />,
    );

    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelector('strong')).toBeNull();
    expect(screen.getByText('脚本链接').closest('a')).toBeNull();
    expect(screen.getByText('数据链接').closest('a')).toBeNull();
    expect(screen.getAllByText(/脚本链接|数据链接/).every((item) =>
      item.classList.contains('markdown-disabled-link'),
    )).toBe(true);
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.getByText('[图片：远程像素]')).toBeInTheDocument();
  });

  it('keeps incomplete streaming syntax renderable until the closing delta arrives', () => {
    const { container, rerender } = render(
      <MarkdownMessage content={'## 流式标题\n\n```ts\nconst answer ='} />,
    );

    expect(screen.getByRole('heading', { name: '流式标题' })).toBeInTheDocument();
    expect(container.querySelector('pre code')).toHaveTextContent('const answer =');

    rerender(
      <MarkdownMessage
        content={'## 流式标题\n\n```ts\nconst answer = 42\n```\n\n| a | b |\n| - | - |\n| 1 |'}
      />,
    );
    expect(container.querySelector('pre code')).toHaveTextContent('const answer = 42');
    expect(screen.getByRole('table')).toBeInTheDocument();
  });
});
