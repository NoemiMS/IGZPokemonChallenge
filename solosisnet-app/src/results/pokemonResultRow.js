import React from 'react';

class PokemonResultsRow extends React.Component {
  render() {
    return (
      <div className="component-pokemon-result-row">
        <span
          className="title">
          {this.props.title}
        </span>
      </div>
    );
  }
}
PokemonResultsRow.propTypes = {
  title: React.PropTypes.string,
};
export default EmojiResultsRow;
