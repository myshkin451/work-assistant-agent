import { memo, type ComponentProps } from 'react';
import ReactMarkdown, { defaultUrlTransform, type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

const remarkPlugins = [remarkGfm];

const components: Components = {
  a({ node: _node, href, ...props }) {
    void _node;
    if (!href) {
      return <span className="markdown-disabled-link">{props.children}</span>;
    }
    const external = typeof href === 'string' && /^https?:\/\//i.test(href);
    return (
      <a
        {...props}
        href={href}
        {...(external ? { target: '_blank', rel: 'noreferrer noopener' } : {})}
      />
    );
  },
  img({ alt }) {
    return alt ? <span className="markdown-image-alt">[图片：{alt}]</span> : null;
  },
  table({ node: _node, ...props }: ComponentProps<'table'> & { node?: unknown }) {
    void _node;
    return (
      <div className="markdown-table-scroll">
        <table {...props} />
      </div>
    );
  },
};

export const MarkdownMessage = memo(function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        components={components}
        remarkPlugins={remarkPlugins}
        skipHtml
        urlTransform={defaultUrlTransform}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
